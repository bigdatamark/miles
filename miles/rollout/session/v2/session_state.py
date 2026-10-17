# [MODULE DOC]
# Serving policy of the v2 session server over the trajectory tree. A request is attached under
# the deepest node whose full path is a prefix of the request (a new root if none); a truncated
# (length-capped) node cannot be extended (conflict error). Provides: the per-session state
# (lock, closing flag, the tree, evaluation flag, sampling defaults, replay flag), attach point
# search with a conflict check, request preparation (resolve args against the parent's recorded
# request, then render prompt tokens by reusing the parent's token snapshot and tokenizing only the
# new suffix), generation commit with the prefix consistency check, and a registry subclass
# creating v2 sessions and computing token mismatch diagnostics. Nothing is ever rejected for
# mismatching history and nothing is deleted. Callers hold the session lock for mutations.
"""Session state over the trajectory tree: always-branch serving.

This module is the serving policy on top. A request is never rejected for mismatching stored history and never
destroys anything: it attaches at the deepest matching node, its unmatched
suffix becomes the new branch's delta, and whatever is not a clean
extension just grows a sibling or a new root. Whether a branch was a retry
is decided later by the sample_picker, not here.

Concurrency contract: single lock on the whole tree. A commit only appends a new
node under the parent the request attached to, so concurrent generations from
the same spot become sibling nodes instead of a conflict.

The tree is the whole state. Serving a request is
``prepare_token_ids_and_request_args`` (``attach_point_for_request``, pure: which
node the request continues; the request args against that node's ``turn_args``;
the prompt rendered under that node) and then ``commit_generation`` under that
node. What ``GET /sessions`` shows as a single chain is
``SessionStateV2.latest()``, the most recently committed generation's path —
derived, never stored.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from miles.rollout.session.config import SessionServerConfig
from miles.rollout.session.errors import MessageValidationError, TokenizationError, TruncatedGenerationError
from miles.rollout.session.linear_trajectory import SessionRegistry, assert_pretokenized_prefix
from miles.rollout.session.request_args import PreparedChatRequest, prepare_chat_request
from miles.rollout.session.types import SessionRecord
from miles.rollout.session.v2.tree_trajectory import AttachPoint, SessionTree, TrajectoryNode
from miles.utils.chat_template_utils.message_matcher_hub import SessionMessageMatcher
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer, extract_template_args

logger = logging.getLogger(__name__)


@dataclass
class SessionStateV2:
    # [CLASS DOC] Mutable state of one v2 session: async lock and closing flag (excluded from
    # repr and comparison), the trajectory tree, evaluation flag, sampling defaults mapping and
    # sampling-support replay flag.
    """Per-session concurrency container plus the trajectory forest.

    The forest is the whole state; ``latest()`` derives the single chain that
    ``GET /sessions`` serves. A failed first turn commits nothing, so the
    session stays fully retryable.
    """

    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)
    closing: bool = field(default=False, repr=False, compare=False)
    tree: SessionTree = field(default_factory=SessionTree)
    evaluation: bool = False
    sampling_defaults: dict[str, Any] = field(default_factory=dict)
    sampling_support_replay: bool = False

    def latest(self) -> TrajectoryNode | None:
        # [FUNCTION DOC] The last node created (highest commit order), or none for an empty tree.
        """The most recently committed generation (always a leaf), or ``None``
        before the first commit. Its root->node path is the single-chain view."""
        return self.tree.nodes[-1] if self.tree.nodes else None


def attach_point_for_request(
    state: SessionStateV2,
    request_messages: list[dict[str, Any]],
    *,
    message_matcher: SessionMessageMatcher | None = None,
) -> AttachPoint:
    # [FUNCTION DOC] Find where the request attaches via the tree search (with the given matcher,
    # strict by default). If the matched node is a truncated generation, raise a truncated
    # generation error (HTTP 409 conflict) since extending it is not allowed. If the attach node
    # is not the latest node, emit a debug log describing the branching. Returns the attach point
    # (node may be none for a new root). Does not mutate the tree.
    """Where *request_messages* attaches: the deepest node whose path is a
    prefix of the request, ``node=None`` for a new root. Pure; raises 409 when
    that node is a truncated generation."""
    attach = state.tree.find_attach_point(request_messages, message_matcher=message_matcher)

    if attach.node is not None and attach.node.truncated:
        raise TruncatedGenerationError(
            "truncated generation cannot be extended: the matched node ended with "
            "finish_reason='length' and truncation closes that path for good; "
            "branch before the cut instead"
        )

    if attach.node is not state.latest():
        logger.debug(
            "Branching: request(%d msgs) attaches at node seq=%s "
            "(matched %d msgs, best overlap %d), tree has %d nodes",
            len(request_messages),
            attach.node.seq if attach.node is not None else "<new root>",
            attach.matched_messages,
            attach.best_overlap,
            len(state.tree.nodes),
        )
    return attach


def prepare_token_ids_and_request_args(
    state: SessionStateV2,
    client_args: dict[str, Any],
    *,
    config: SessionServerConfig,
    tito_tokenizer: TITOTokenizer,
    message_matcher: SessionMessageMatcher | None = None,
) -> tuple[PreparedChatRequest, TrajectoryNode | None]:
    # [FUNCTION DOC] Resolve the attach node (the parent) from the request messages, prepare the
    # request arguments against the parent's recorded request (none for a new root) using the
    # session's evaluation flag, sampling defaults and replay flag, then render prompt token ids
    # and store them in the prepared body under input ids. Returns the prepared request and the
    # parent node (none for a new root) for the later commit. Does not mutate the tree; errors
    # (conflict, validation) propagate.
    """Return the prepared request with `body["input_ids"]` and its parent for `commit_generation`."""
    request_messages = client_args.get("messages", [])
    parent = attach_point_for_request(state, request_messages, message_matcher=message_matcher).node
    prepared = prepare_chat_request(
        client_args,
        tito_tokenizer,
        config=config,
        turn_args=parent.turn_args if parent is not None else None,
        evaluation=state.evaluation,
        sampling_defaults=state.sampling_defaults,
        sampling_support_replay=state.sampling_support_replay,
    )
    prepared.body["input_ids"] = _render_token_ids(
        parent, request_messages, template_args=prepared.template_args, tito_tokenizer=tito_tokenizer
    )
    return prepared, parent


def _render_token_ids(
    parent: TrajectoryNode | None,
    request_messages: list[dict[str, Any]],
    *,
    template_args: dict[str, Any],
    tito_tokenizer: TITOTokenizer,
) -> list[int]:
    # [FUNCTION DOC] HELPER: Return prompt token ids. No parent: tokenize all request messages
    # with the chat template (generation prompt added). With a parent: the stored messages are the
    # messages along the parent's path; the request's messages beyond that count are the suffix,
    # whose roles must be allowed (message validation error otherwise); the effective new message
    # list is the stored messages plus the suffix; the parent's token snapshot is reused as the
    # pretokenized prefix and only the suffix is newly tokenized. Note the request's own spelling
    # of the prefix messages is not checked here (attach search already matched them).
    """Build prompt token IDs for a request extending `parent`.

    Use `template_args` for newly rendered tokens.

    - No parent (new root): render the whole request from scratch.
    - Otherwise: reuse the parent's token snapshot as-is and tokenize only
      the new suffix on top — the shared prefix is never re-rendered.
    """
    if parent is None:
        return tito_tokenizer.apply_chat_template(
            request_messages,
            add_generation_prompt=True,
            tokenize=True,
            template_args=template_args,
        )

    stored = parent.path_messages()
    _validate_suffix_roles(request_messages[len(stored) :], tito_tokenizer)
    effective_messages = stored + request_messages[len(stored) :]
    return tito_tokenizer.merge_tokens(
        old_messages=stored,
        new_messages=effective_messages,
        pretokenized_token_ids=parent.token_ids,
        template_args=template_args,
    )


def _validate_suffix_roles(
    suffix: list[dict[str, Any]],
    tito_tokenizer: TITOTokenizer,
) -> None:
    # [FUNCTION DOC] HELPER: Raise a message validation error for the first appended message whose
    # role (missing role counts as none) is not among the tokenizer's allowed append roles; the
    # message lists allowed roles sorted.
    allowed = set(tito_tokenizer.allowed_append_roles)
    for message in suffix:
        role = message.get("role")
        if role not in allowed:
            raise MessageValidationError(
                f"appended message role={role!r} not allowed "
                f"(allowed={sorted(set(tito_tokenizer.allowed_append_roles))}); "
                "the selected TITO fixed template does not support appending this role"
            )


def commit_generation(
    state: SessionStateV2,
    *,
    parent: TrajectoryNode | None,
    request_messages: list[dict[str, Any]],
    assistant_message: dict[str, Any],
    prompt_token_ids: list[int],
    completion_token_ids: list[int],
    max_trim_tokens: int,
    record: SessionRecord,
    response_id: str,
    finish_reason: str,
    turn_args: dict[str, Any] | None = None,
) -> TrajectoryNode:
    # [FUNCTION DOC] Commit a generation as a new tree node and return it (caller holds the lock).
    # Validates that the parent's token snapshot (empty for a new root) is a prefix of prompt plus
    # completion ids within the trim tolerance (tokenization error otherwise, nothing added).
    # The node's delta messages are the request messages beyond the parent's path messages plus
    # the assistant message; its tokens are prompt plus completion ids; its completion span runs
    # from the prompt length to the total length; commit time is the record's timestamp. The tree
    # enforces its node cap (value error when exceeded).
    """Validate and append one generation under *parent* (the request's attach
    node). Prefix validation is byte-identical to the pre-tree checkpoint check.
    ``turn_args`` is the full resolved request for this generation."""
    all_token_ids = prompt_token_ids + completion_token_ids
    assert_pretokenized_prefix(
        parent.token_ids if parent is not None else [],
        all_token_ids,
        max_trim_tokens=max_trim_tokens,
        request_messages=request_messages,
        assistant_message=assistant_message,
    )

    parent_messages = parent.path_messages() if parent is not None else []
    delta = list(request_messages[len(parent_messages) :]) + [assistant_message]
    node = state.tree.create_node(
        parent,
        delta_messages=delta,
        token_ids=all_token_ids,
        completion_span=(len(prompt_token_ids), len(all_token_ids)),
        committed_at=record.timestamp,
        response_id=response_id,
        record=record,
        finish_reason=finish_reason,
        turn_args=turn_args,
    )
    return node


class SessionRegistryV2(SessionRegistry):
    # [CLASS DOC] Session registry whose sessions are v2 states: reuses all base resources and
    # lookup behavior, overriding session creation and mismatch computation (the base per-session
    # mismatch method is replaced by one over the latest generation's path).
    """Session ID -> session state mapping with shared tokenizer resources.

    The v1 registry shell (CRUD + tokenizer resources) with the session type
    swapped to ``SessionStateV2``; all session mutations go through the
    module-level serving functions, called by the route handler under
    ``SessionStateV2.lock``.
    """

    sessions: dict[str, SessionStateV2]

    def create_session(
        self,
        *,
        evaluation: bool = False,
        sampling_defaults: dict[str, Any] | None = None,
        sampling_support_replay: bool = False,
    ) -> str:
        # [FUNCTION DOC] Create a v2 state with the evaluation flag, a copy of the sampling
        # defaults (empty if none) and the replay flag, register it under a new random 32-character
        # hex id, and return the id.
        session_id = uuid.uuid4().hex
        self.sessions[session_id] = SessionStateV2(
            evaluation=evaluation,
            sampling_defaults=dict(sampling_defaults or {}),
            sampling_support_replay=sampling_support_replay,
        )
        return session_id

    def compute_mismatch(
        self,
        messages: list[dict[str, Any]],
        token_ids: list[int],
        *,
        turn_args: dict[str, Any],
    ) -> list[dict] | None:
        # [FUNCTION DOC] Read-only diagnostic for one path. Returns none for empty token ids.
        # Renders the messages through the chat template (no generation prompt, tokenized, with
        # the renderer fields from the recorded turn args) and compares with the given tokens using
        # the comparator; returns mismatches as dictionaries (empty list if identical). Any
        # exception is re-raised as a tokenization error.
        """Compare accumulated token IDs against canonical chat template
        output for one path using its leaf's recorded renderer fields. Read-only."""
        if not token_ids:
            return None
        try:
            # Mismatch checks have no request args to resolve. Re-render from
            # the renderer fields in the full request recorded for this turn.
            expected_ids = self.tito_tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=True,
                template_args=extract_template_args(turn_args),
            )
            mismatches = self.comparator.compare_sequences(expected_ids, token_ids)
            return [m.to_dict() for m in mismatches]
        except Exception as e:
            raise TokenizationError(f"failed to compute tito_session_mismatch: {e}") from e

    def compute_session_mismatch(self, state: SessionStateV2) -> list[dict] | None:
        # [FUNCTION DOC] Mismatch diagnostic for the latest committed generation: none if the tree
        # is empty, else the per-path comparison using that node's path messages, token ids and
        # recorded request args.
        """``compute_mismatch`` over the latest committed generation's path."""
        node = state.latest()
        if node is None:
            return None
        return self.compute_mismatch(node.path_messages(), node.token_ids, turn_args=node.turn_args)
