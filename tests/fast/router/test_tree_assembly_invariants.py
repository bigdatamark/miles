# [MODULE DOC] Property/fuzz tests for the v2 sample assembly over a tree (forest) of committed
# generations. Synthetic trees are grown with globally unique token ids, assembled through the v2
# core's sample collection, and the decoded samples are checked against an independent oracle:
# which leaves are kept (rolled-back leaves dropped), ownership of shared spans (each completion
# span is trained exactly once, in the earliest surviving leaf containing it), prefix legality of
# tokens, mask and log-prob lengths, reward propagation, ordering, and truncation behavior.
# Targeted edge cases (chained rollbacks, equal-length twins, nested forks, mid-path length turn)
# complete the module. The module-level argument namespace configures the Qwen3 0.6B tokenizer
# and the rolled-back-leaf picker; it requires the tokenizer to be loadable.
"""Tree-merge invariants for randomized forests and targeted edges.

The independent oracle checks:

1. token fields match each possibly truncated leaf snapshot;
2. every kept completion span is trainable in exactly one surviving leaf;
3. rolled-back leaf trimming accepts shorter replacements;
4. one trajectory reward reaches every kept sample;
5. samples sort by checkpoint count, then leaf seq, descending.
"""

import json
import random
import uuid
from types import SimpleNamespace

import pytest
from tests.fast.rollout.session.test_samples import _make_record

from miles.rollout.session.samples.codec import COMPUTED_FIELDS_V2, decode_samples_and_merge_input_sample
from miles.rollout.session.v2.core import SessionCoreV2
from miles.rollout.session.v2.session_state import SessionRegistryV2
from miles.utils.chat_template_utils import get_tito_tokenizer
from miles.utils.processing_utils import load_tokenizer
from miles.utils.types import Sample

_ARGS = SimpleNamespace(
    miles_router_timeout=30,
    hf_checkpoint="Qwen/Qwen3-0.6B",
    chat_template_path=None,
    apply_chat_template_kwargs={"enable_thinking": False},
    tito_model="default",
    sglang_speculative_algorithm=None,
    use_rollout_routing_replay=False,
    use_rollout_indexer_replay=False,
    use_sampling_support_replay=False,
    lora_rank=0,
    lora_adapter_path=None,
    lora_train_only=False,
    session_server_instance_id=uuid.uuid4().hex,
    save_debug_trajectory_data=None,
    session_sample_picker_path="miles.rollout.session.v2.picker_hub.drop_rolled_back_leaves",
    session_sample_postprocessor_path="miles.rollout.session.v2.postprocessor_hub.default_postprocess",
)


class _UnusedBackend:
    # [CLASS DOC] HELPER: Stand-in proxy backend that fails if used, proving sample collection
    # never touches the backend.
    async def do_proxy(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Async; ignores all arguments and always raises an assertion error
        # saying collect_samples must not touch the proxy backend.
        raise AssertionError("collect_samples must not touch the proxy backend")


@pytest.fixture(scope="module")
def core():
    # [FUNCTION DOC] HELPER: Module-scoped fixture. Loads the Qwen3 0.6B tokenizer (trusting
    # remote code, no custom template), builds the TITO tokenizer with the default model type and
    # the thinking-disabled kwargs, creates a v2 session registry and returns a v2 session core
    # over the unused backend, the registry, the argument namespace and the fixed instance id.
    # Performs tokenizer loading (may touch disk or network).
    tokenizer = load_tokenizer(_ARGS.hf_checkpoint, chat_template_path=None, trust_remote_code=True)
    tito_tokenizer = get_tito_tokenizer(
        tokenizer,
        tokenizer_type=_ARGS.tito_model,
        chat_template_kwargs=_ARGS.apply_chat_template_kwargs,
    )
    registry = SessionRegistryV2(tokenizer, tito_tokenizer=tito_tokenizer)
    return SessionCoreV2(_UnusedBackend(), registry, _ARGS, _ARGS.session_server_instance_id)


class _Grower:
    # [CLASS DOC] HELPER: Builds tree nodes in a session state from synthetic records. Holds the
    # state and a running counter so every token id it hands out is globally unique, which means
    # no two roots share a prompt.
    """Grows legal trees out of synthetic records with globally unique token ids."""

    def __init__(self, state):
        # [FUNCTION DOC] HELPER: Stores the session state and starts the token counter at 1.
        self.state = state
        self.next_token = 1

    def _fresh(self, n: int) -> list[int]:
        # [FUNCTION DOC] HELPER: Returns the next n consecutive unused token ids and advances the
        # counter by n.
        ids = list(range(self.next_token, self.next_token + n))
        self.next_token += n
        return ids

    def grow(self, parent, *, env_len: int, completion_len: int, finish_reason: str = "stop"):
        # [FUNCTION DOC] HELPER: Adds one node under the given parent (or a new root when none).
        # The prompt is the parent's full token ids plus env_len fresh ids; the completion is
        # completion_len fresh ids. Builds a synthetic record for them with the finish reason
        # (default stop) and creates the node with no delta messages, token ids equal to prompt
        # plus completion, the completion span covering exactly the completion part, a commit time
        # equal to the current node count, a response id derived from the node count, the record
        # and the finish reason. Returns the new node.
        prefix = parent.token_ids if parent is not None else []
        prompt = prefix + self._fresh(env_len)
        completion = self._fresh(completion_len)
        record = _make_record(prompt_token_ids=prompt, output_token_ids=completion, finish_reason=finish_reason)
        return self.state.tree.create_node(
            parent,
            delta_messages=[],
            token_ids=prompt + completion,
            completion_span=(len(prompt), len(prompt) + len(completion)),
            committed_at=float(len(self.state.tree.nodes)),
            response_id=f"resp-{len(self.state.tree.nodes)}",
            record=record,
            finish_reason=finish_reason,
        )


async def _fresh_grower(core):
    # [FUNCTION DOC] HELPER: Async; creates a session through the core, parses the session id from
    # the response body, looks up the session state in the registry, and returns the id, the state
    # and a new grower over it.
    response = await core.create_session()
    sid = json.loads(response.body)["session_id"]
    state = core.registry.sessions[sid]
    return sid, state, _Grower(state)


def _oracle_pick(state):
    # [FUNCTION DOC] HELPER: Oracle for which leaves survive. Root leaves are always kept; a
    # non-root leaf is trimmed when its parent has a child with a larger sequence number (a later
    # sibling), otherwise kept. Returns (kept, trimmed), with kept sorted by path length then
    # sequence number, both descending.
    """Independent re-derivation of drop_rolled_back_leaves: (kept, trimmed).

    Every root leaf is kept: the grower's token ids are globally unique, so no
    two roots share a prompt and the re-sent-first-turn rule never applies.
    """
    kept, trimmed = [], []
    for leaf in state.tree.leaves():
        if leaf.parent is None:
            kept.append(leaf)
            continue
        later = [s for s in leaf.parent.children if s.seq > leaf.seq]
        if not later:
            kept.append(leaf)
            continue
        trimmed.append(leaf)
    kept.sort(key=lambda leaf: (len(leaf.path_nodes()), leaf.seq), reverse=True)
    return kept, trimmed


def _oracle_owner(kept):
    # [FUNCTION DOC] HELPER: Oracle for span ownership. Visits kept leaves in ascending sequence
    # order and assigns each node on a leaf's path to the first leaf that contains it. Returns a
    # mapping from node sequence number to owning leaf sequence number.
    owner = {}
    for leaf in sorted(kept, key=lambda leaf: leaf.seq):
        for node in leaf.path_nodes():
            owner.setdefault(node.seq, leaf.seq)
    return owner


def _assert_sample_legality(sample: Sample, leaf, owner: dict[int, int]) -> None:
    # [FUNCTION DOC] HELPER: Asserts that a decoded sample is legal for its leaf: tokens are a
    # prefix of the leaf's token snapshot; the loss mask and rollout log-prob lists have the
    # response length; and the set of positions with mask one equals the union of completion
    # spans of path nodes owned by this leaf, clipped to the sample's response range (so an
    # early-stopped sample shorter than the snapshot is tolerated). Raises assertion errors with
    # explanatory messages.
    snapshot = leaf.token_ids
    assert sample.tokens == snapshot[: len(sample.tokens)], "sample tokens must be a prefix of the leaf snapshot"
    assert len(sample.loss_mask) == sample.response_length
    assert len(sample.rollout_log_probs) == sample.response_length

    response_start = len(sample.tokens) - sample.response_length
    trained = {response_start + i for i, bit in enumerate(sample.loss_mask) if bit == 1}
    expected = set()
    for node in leaf.path_nodes():
        if owner[node.seq] != leaf.seq:
            continue
        lo, hi = node.completion_span
        expected |= set(range(lo, min(hi, len(sample.tokens))))
    # Mid-path early-stop (a non-COMPLETED turn ends the fold) may keep the
    # sample shorter than the snapshot; trained positions must then be the
    # owned completions clipped to the kept range.
    expected = {p for p in expected if response_start <= p < len(sample.tokens)}
    assert trained == expected, f"exactly-once violated: trained={sorted(trained)} expected={sorted(expected)}"


async def _collect(core, sid, *, max_seq_len=None, agent_metadata=None):
    # [FUNCTION DOC] HELPER: Async; asks the core to collect samples for the session with the
    # optional maximum sequence length and agent metadata, and returns the response's status code
    # and body bytes.
    response = await core.collect_samples(sid, max_seq_len=max_seq_len, agent_metadata=agent_metadata)
    return response.status_code, bytes(response.body)


def _grow_random_tree(grower, rng: random.Random, *, allow_shorter_replacements: bool = False):
    # [FUNCTION DOC] HELPER: Grows a random forest with the given random generator: first a root
    # (environment 1-3 tokens, completion 1-4), then 2-9 random operations: with probability 0.15 a
    # new root; up to probability 0.45 a rollback, picking a random leaf and, for a non-root,
    # adding a sibling under its parent with a completion length of at least the abandoned
    # length (or at least 1 when shorter replacements are allowed) and up to two more, while a
    # root leaf is simply extended; otherwise extend a random leaf (environment 1-2, completion
    # 1-3). Mutates the grower's state; returns nothing.
    """Grow a random forest, optionally allowing shorter replacement branches."""
    grower.grow(None, env_len=rng.randint(1, 3), completion_len=rng.randint(1, 4))
    for _ in range(rng.randint(2, 9)):
        state = grower.state
        op = rng.random()
        leaves = state.tree.leaves()
        if op < 0.15:  # new root (subagent)
            grower.grow(None, env_len=rng.randint(1, 3), completion_len=rng.randint(1, 4))
        elif op < 0.45:  # rollback: supersede a random childless leaf with a sibling
            leaf = rng.choice(leaves)
            if leaf.parent is None:
                grower.grow(leaf, env_len=rng.randint(1, 2), completion_len=rng.randint(1, 3))
                continue
            abandoned_len = len(leaf.token_ids) - len(leaf.parent.token_ids)
            floor = 1 if allow_shorter_replacements else max(1, abandoned_len)
            grower.grow(
                leaf.parent,
                env_len=rng.randint(1, 2),
                completion_len=max(floor, rng.randint(1, abandoned_len + 2)),
            )
        else:  # extend a random leaf
            grower.grow(rng.choice(leaves), env_len=rng.randint(1, 2), completion_len=rng.randint(1, 3))


@pytest.mark.parametrize("seed", range(25))
async def test_fuzz_forest_assembly_invariants(core, seed):
    # [FUNCTION DOC] For 25 seeds: grows a random forest, computes the oracle kept and trimmed
    # leaves, collects samples with a random trajectory reward in the agent metadata, and asserts
    # status 200, one sample per kept leaf in the oracle order, each sample's leaf id matching,
    # legality per sample, the reward on every sample, and that no trimmed leaf is emitted.
    sid, state, grower = await _fresh_grower(core)
    rng = random.Random(seed)
    _grow_random_tree(grower, rng)

    kept, trimmed = _oracle_pick(state)
    trajectory_reward = round(rng.random(), 3)

    status, payload = await _collect(core, sid, agent_metadata={"reward": trajectory_reward})
    assert status == 200
    reply = decode_samples_and_merge_input_sample(payload, Sample(), fields=COMPUTED_FIELDS_V2)
    assert len(reply.samples) == len(kept)
    owner = _oracle_owner(kept)
    for sample, leaf in zip(reply.samples, kept, strict=True):
        assert sample.metadata["leaf"]["node_id"] == leaf.seq
        _assert_sample_legality(sample, leaf, owner)
        assert sample.reward == trajectory_reward
    emitted = {s.metadata["leaf"]["node_id"] for s in reply.samples}
    assert emitted.isdisjoint({leaf.seq for leaf in trimmed})


@pytest.mark.parametrize("seed", range(50, 70))
async def test_fuzz_shorter_replacements_assemble_legally(core, seed):
    # [FUNCTION DOC] For seeds 50-69: grows a forest allowing shorter replacement branches,
    # collects samples, and asserts status 200, one sample per oracle-kept leaf, and legality of
    # each sample against the oracle ownership.
    """Shorter replacements still assemble under the temporal trim rule."""
    sid, state, grower = await _fresh_grower(core)
    rng = random.Random(seed)
    _grow_random_tree(grower, rng, allow_shorter_replacements=True)

    kept, _ = _oracle_pick(state)
    status, payload = await _collect(core, sid)
    assert status == 200
    reply = decode_samples_and_merge_input_sample(payload, Sample(), fields=COMPUTED_FIELDS_V2)
    assert len(reply.samples) == len(kept)
    owner = _oracle_owner(kept)
    for sample, leaf in zip(reply.samples, kept, strict=True):
        _assert_sample_legality(sample, leaf, owner)


@pytest.mark.parametrize("seed", range(25, 35))
async def test_fuzz_forest_invariants_under_truncation(core, seed):
    # [FUNCTION DOC] For seeds 25-34: grows a forest and collects samples with a random maximum
    # sequence length of 4-12 tokens. If no samples return, the empty reason must be all
    # truncated. Otherwise the emitted leaves are looked up among the kept ones, ownership is
    # recomputed over just those leaves, and every sample must be no longer than the maximum and
    # legal.
    sid, state, grower = await _fresh_grower(core)
    rng = random.Random(seed)
    _grow_random_tree(grower, rng)

    kept, _ = _oracle_pick(state)
    max_seq_len = rng.randint(4, 12)

    status, payload = await _collect(core, sid, max_seq_len=max_seq_len)
    assert status == 200
    reply = decode_samples_and_merge_input_sample(payload, Sample(), fields=COMPUTED_FIELDS_V2)
    if not reply.samples:
        assert reply.empty_reason == "all_truncated"
        return
    # Leaves whose every turn truncated away drop from the material; the
    # survivors' ownership is recomputed over what remains (no orphan spans).
    emitted_ids = [s.metadata["leaf"]["node_id"] for s in reply.samples]
    emitted_leaves = [next(leaf for leaf in kept if leaf.seq == node_id) for node_id in emitted_ids]
    owner = _oracle_owner(emitted_leaves)
    for sample, leaf in zip(reply.samples, emitted_leaves, strict=True):
        assert len(sample.tokens) <= max_seq_len
        _assert_sample_legality(sample, leaf, owner)


class TestTargetedEdges:
    # [CLASS DOC] Hand-built tree shapes with exact expectations.
    async def test_chained_rollbacks_single_survivor(self, core):
        # [FUNCTION DOC] A root with four children added in order (three abandoned, one last):
        # collection yields exactly one sample, for the last child.
        sid, state, grower = await _fresh_grower(core)
        root = grower.grow(None, env_len=2, completion_len=2)
        for _ in range(3):  # three abandoned attempts, each superseded
            grower.grow(root, env_len=1, completion_len=2)
        survivor = grower.grow(root, env_len=1, completion_len=2)

        status, payload = await _collect(core, sid)
        assert status == 200
        reply = decode_samples_and_merge_input_sample(payload, Sample(), fields=COMPUTED_FIELDS_V2)
        (sample,) = reply.samples
        assert sample.metadata["leaf"]["node_id"] == survivor.seq

    async def test_twins_equal_length_earlier_trimmed(self, core):
        # [FUNCTION DOC] Two equal-length sibling children of a root: only the later twin yields a
        # sample (the earlier one is trimmed by the temporal rule).
        sid, state, grower = await _fresh_grower(core)
        root = grower.grow(None, env_len=2, completion_len=2)
        grower.grow(root, env_len=1, completion_len=2)
        twin_b = grower.grow(root, env_len=1, completion_len=2)

        status, payload = await _collect(core, sid)
        assert status == 200
        reply = decode_samples_and_merge_input_sample(payload, Sample(), fields=COMPUTED_FIELDS_V2)
        (sample,) = reply.samples  # temporal rule: the earlier equal-length twin is trimmed
        assert sample.metadata["leaf"]["node_id"] == twin_b.seq

    async def test_fork_below_fork_ownership_nesting(self, core):
        # [FUNCTION DOC] Builds a root, a middle node, a deep chain below it and later siblings
        # below the middle and the root. After collection checks legality for every kept sample,
        # then cross-checks that, over all samples, each surviving node's completion span is fully
        # trained in exactly one sample.
        """Two nested fork points: each shared span trains exactly once, in
        the earliest surviving leaf that contains it."""
        sid, state, grower = await _fresh_grower(core)
        root = grower.grow(None, env_len=2, completion_len=2)
        mid = grower.grow(root, env_len=1, completion_len=2)
        deep_a = grower.grow(mid, env_len=1, completion_len=2)  # earliest leaf: owns root+mid+own
        grower.grow(deep_a, env_len=1, completion_len=2)  # extend: deep_a no longer a leaf
        grower.grow(mid, env_len=1, completion_len=3)  # later sibling below mid
        grower.grow(root, env_len=1, completion_len=4)  # sibling below root

        kept, _ = _oracle_pick(state)
        status, payload = await _collect(core, sid)
        assert status == 200
        reply = decode_samples_and_merge_input_sample(payload, Sample(), fields=COMPUTED_FIELDS_V2)
        assert len(reply.samples) == len(kept)
        owner = _oracle_owner(kept)
        for sample, leaf in zip(reply.samples, kept, strict=True):
            _assert_sample_legality(sample, leaf, owner)
        # Cross-check: the union of trained spans over all samples covers each
        # surviving-path completion exactly once.
        span_train_count: dict[int, int] = {}
        for sample, leaf in zip(reply.samples, kept, strict=True):
            response_start = len(sample.tokens) - sample.response_length
            trained = {response_start + i for i, bit in enumerate(sample.loss_mask) if bit == 1}
            for node in leaf.path_nodes():
                lo, hi = node.completion_span
                if set(range(lo, hi)) <= trained:
                    span_train_count[node.seq] = span_train_count.get(node.seq, 0) + 1
        surviving_nodes = {node.seq for leaf in kept for node in leaf.path_nodes()}
        assert span_train_count == {seq: 1 for seq in surviving_nodes}

    async def test_midpath_length_turn_early_stops_but_stays_legal(self, core):
        # [FUNCTION DOC] A chain root, a middle node finished by length, and a leaf: the single
        # sample stops at the length turn (token count equals the middle node's snapshot length)
        # and is still legal for the leaf.
        """A non-COMPLETED mid-path turn ends the fold (documented early-stop):
        the sample stops at that turn and every trained position still lies in
        an owned completion span."""
        sid, state, grower = await _fresh_grower(core)
        root = grower.grow(None, env_len=2, completion_len=2)
        cut = grower.grow(root, env_len=1, completion_len=2, finish_reason="length")
        leaf = grower.grow(cut, env_len=1, completion_len=2)

        status, payload = await _collect(core, sid)
        assert status == 200
        reply = decode_samples_and_merge_input_sample(payload, Sample(), fields=COMPUTED_FIELDS_V2)
        (sample,) = reply.samples
        assert len(sample.tokens) == len(cut.token_ids)  # fold stopped at the length turn
        _assert_sample_legality(sample, leaf, _oracle_owner([leaf]))
