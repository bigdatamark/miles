# [MODULE DOC] Async tests of the agentic tool-call generate function when run against the v2
# session server. A fake tracer (returning a canned samples reply, or raising a configured error)
# replaces the real endpoint tracer's factory, and the agent function loader is replaced by a stub
# agent returning a fixed result. Covered: returned sample list and forwarded agent metadata,
# propagation of evaluation flag and sampling parameters to the tracer, assignment of a shared
# rollout id to leaf samples (taken from the input sample's rollout id or index), aborted-sample
# fallbacks for empty replies and transport errors, replacement of stale session metrics with the
# server's authoritative metrics, validation errors for missing, mismatched or unavailable
# metrics, rejection of missing session server instances, and propagation of other collection
# errors.
from types import SimpleNamespace

import pytest

import miles.rollout.generate_hub.agentic_tool_call as agentic_tool_call
from miles.ray.rollout.rollout_data_conversion import validate_compact_rollout_ids
from miles.rollout.base_types import GenerateFnInput
from miles.rollout.session.samples.codec import SamplesReply
from miles.rollout.session.types import SessionServerInstance
from miles.rollout.session.v2.metrics import SESSION_ROLLOUT_METRICS_KEY
from miles.utils.types import Sample


class _Tracer:
    # [CLASS DOC] HELPER: Fake endpoint tracer. Class attributes give a fixed session id, server
    # address, no server instance id and base URLs (the agent base URL equals the base URL). An
    # instance holds a canned reply or an error to raise, plus the agent metadata passed to it and
    # (set externally by the patched factory) the evaluation flag and sampling parameters.
    session_id = "sid-1"
    session_server_id = "127.0.0.1:12345"
    session_server_instance_id = None
    base_url = "http://127.0.0.1:12345/sessions/sid-1"
    agent_base_url = base_url

    def __init__(self, reply=None, error=None):
        # [FUNCTION DOC] HELPER: Stores the reply and error (both default none) and initializes
        # the recorded agent metadata to none.
        self.reply = reply
        self.error = error
        self.agent_metadata = None

    async def collect_samples(self, input_sample, *, max_seq_len, agent_metadata=None):
        # [FUNCTION DOC] HELPER: Records the agent metadata, raises the configured error if one is
        # set, otherwise returns the canned reply. The input sample and maximum sequence length
        # are ignored.
        self.agent_metadata = agent_metadata
        if self.error is not None:
            raise self.error
        return self.reply


def _generate_input(*, evaluation=False, sampling_params=None, **args_kwargs) -> GenerateFnInput:
    # [FUNCTION DOC] HELPER: Builds the input for the generate function. Arguments namespace
    # defaults: one session server instance at a local address, a fake agent function path, inline
    # agent mode, no maximum sequence length, partial rollout off, session server version v2, and
    # no speculative algorithm; extra keyword arguments override or extend these. The sample has
    # group index 3, index 7, a one-message user prompt, a label and metadata with a source entry.
    # The state wraps only the arguments. Sampling parameters default to an empty mapping.
    args = SimpleNamespace(
        **{
            "session_server_instances": [SessionServerInstance(addr="127.0.0.1:12345")],
            "custom_agent_function_path": "test.fake_agent",
            "custom_agent_function_mode": "inline",
            "max_seq_len": None,
            "partial_rollout": False,
            "use_session_server": "v2",
            "sglang_speculative_algorithm": None,
            **args_kwargs,
        }
    )
    state = SimpleNamespace(args=args)
    sample = Sample(
        group_index=3,
        index=7,
        prompt=[{"role": "user", "content": "hello"}],
        label="label",
        metadata={"source": "test"},
    )
    return GenerateFnInput(state=state, sample=sample, sampling_params=sampling_params or {}, evaluation=evaluation)


async def _fake_agent(**kwargs):
    # [FUNCTION DOC] HELPER: Stub agent function that ignores its keyword arguments and returns a
    # mapping with agent_result set to "done".
    return {"agent_result": "done"}


def _session_metadata(spec_info=None):
    # [FUNCTION DOC] HELPER: Builds session metadata keyed by the session rollout metrics key,
    # holding session id "sid-1" and metrics with a speculative-decoding info entry (the given
    # mapping, or the default empty speculative info converted to a mapping).
    return {
        SESSION_ROLLOUT_METRICS_KEY: {
            "session_id": "sid-1",
            "metrics": {"spec_info": spec_info or Sample.SpecInfo().to_dict()},
        }
    }


def _patch_agent(monkeypatch, tracer):
    # [FUNCTION DOC] HELPER: Patches the module under test so its tracer factory returns the given
    # tracer (after recording the evaluation flag and sampling parameters on it) and its function
    # loader returns the stub agent regardless of the path.
    async def fake_create(args, *, evaluation=False, sampling_params=None):
        # [FUNCTION DOC] HELPER: Replacement tracer factory; records the arguments on the tracer
        # and returns it.
        tracer.evaluation = evaluation
        tracer.sampling_params = sampling_params
        return tracer

    monkeypatch.setattr(agentic_tool_call.OpenAIEndpointTracer, "create", fake_create)
    monkeypatch.setattr(agentic_tool_call, "load_function", lambda path: _fake_agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("evaluation", [False, True])
async def test_success_returns_list_and_forwards_agent_metadata(monkeypatch, evaluation):
    # [FUNCTION DOC] Parametrized over the evaluation flag. With a tracer returning one completed
    # sample, runs generate with temperature and max new tokens in the sampling params. Asserts the
    # tracer saw the evaluation flag and the sampling params unchanged, the output samples are the
    # returned sample whose rollout id remains unset, and the tracer received the stub agent's
    # result as agent metadata.
    sample = Sample(status=Sample.Status.COMPLETED, response="done", response_length=1, tokens=[1])
    tracer = _Tracer(SamplesReply(samples=[sample], session_metadata={}, empty_reason=None))
    _patch_agent(monkeypatch, tracer)

    generate_input = _generate_input(evaluation=evaluation, sampling_params={"temperature": 0.7, "max_new_tokens": 8})
    output = await agentic_tool_call.generate(generate_input)

    assert tracer.evaluation is evaluation
    # The session fills these into requests the agent sends without them.
    assert tracer.sampling_params == {"temperature": 0.7, "max_new_tokens": 8}
    assert output.samples == [sample]
    assert output.samples[0].rollout_id is None
    assert tracer.agent_metadata == {"agent_result": "done"}


@pytest.mark.asyncio
@pytest.mark.parametrize(("input_rollout_id", "expected_rollout_id"), [(None, 7), (11, 11)])
async def test_success_assigns_shared_rollout_id_to_v2_leaves(monkeypatch, input_rollout_id, expected_rollout_id):
    # [FUNCTION DOC] Parametrized: an input without a rollout id expects the sample's index (7) to
    # be used; an input with rollout id 11 expects 11. With two returned leaf samples, asserts both
    # leaves get the same expected rollout id and the compact rollout id validator accepts the
    # result.
    leaves = [
        Sample(status=Sample.Status.COMPLETED, response="one", response_length=1, tokens=[1]),
        Sample(status=Sample.Status.COMPLETED, response="two", response_length=1, tokens=[2]),
    ]
    tracer = _Tracer(SamplesReply(samples=leaves, session_metadata={}, empty_reason=None))
    _patch_agent(monkeypatch, tracer)
    generate_input = _generate_input()
    generate_input.sample.rollout_id = input_rollout_id

    output = await agentic_tool_call.generate(generate_input)

    assert [sample.rollout_id for sample in output.samples] == [expected_rollout_id] * 2
    validate_compact_rollout_ids([[output.samples]])


@pytest.mark.asyncio
async def test_v2_requires_input_rollout_identity(monkeypatch):
    # [FUNCTION DOC] Clears both the input sample's index and rollout id and asserts generate
    # raises an assertion error saying it requires an input rollout id or index.
    leaves = [
        Sample(status=Sample.Status.COMPLETED, response="one", response_length=1, tokens=[1]),
        Sample(status=Sample.Status.COMPLETED, response="two", response_length=1, tokens=[2]),
    ]
    tracer = _Tracer(SamplesReply(samples=leaves, session_metadata={}, empty_reason=None))
    _patch_agent(monkeypatch, tracer)
    generate_input = _generate_input()
    generate_input.sample.index = None
    generate_input.sample.rollout_id = None

    with pytest.raises(AssertionError, match="require input Sample.rollout_id or Sample.index"):
        await agentic_tool_call.generate(generate_input)


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_reason", ["no_records", "all_truncated"])
async def test_empty_reply_returns_aborted_list(monkeypatch, empty_reason):
    # [FUNCTION DOC] Parametrized over the empty reasons no records and all truncated. With a
    # reply holding no samples, asserts the output is a one-element list containing a new sample
    # (not the input sample) with aborted status and without the session metrics entry in its
    # metadata.
    tracer = _Tracer(SamplesReply(samples=[], session_metadata={}, empty_reason=empty_reason))
    _patch_agent(monkeypatch, tracer)
    generate_input = _generate_input()

    output = await agentic_tool_call.generate(generate_input)

    assert isinstance(output.samples, list)
    assert len(output.samples) == 1
    assert output.samples[0] is not generate_input.sample
    assert output.samples[0].status == Sample.Status.ABORTED
    assert SESSION_ROLLOUT_METRICS_KEY not in output.samples[0].metadata


@pytest.mark.asyncio
async def test_transport_collection_error_has_no_metrics_owner(monkeypatch):
    # [FUNCTION DOC] The tracer raises a timeout error while the input sample's metadata carries a
    # stale session metrics entry. Asserts generate does not propagate it but returns a single
    # aborted sample whose metadata no longer has the session metrics entry.
    tracer = _Tracer(error=TimeoutError("samples unavailable"))
    _patch_agent(monkeypatch, tracer)
    generate_input = _generate_input(sglang_speculative_algorithm="EAGLE")
    generate_input.sample.metadata[SESSION_ROLLOUT_METRICS_KEY] = {"session_id": "stale", "metrics": {}}

    output = await agentic_tool_call.generate(generate_input)

    (sample,) = output.samples
    assert sample.status == Sample.Status.ABORTED
    assert SESSION_ROLLOUT_METRICS_KEY not in sample.metadata


@pytest.mark.asyncio
async def test_v2_replaces_stale_metrics_with_shared_authoritative_carrier(monkeypatch):
    # [FUNCTION DOC] Two leaf samples carry planted stale session metrics; the reply's session
    # metadata carries authoritative metrics with speculative-decoding counters (with the EAGLE
    # algorithm configured). Asserts every output sample's session metrics equal the server's
    # session id and metrics, replacing the planted values.
    stale = {"session_id": "stale", "metrics": {"agent": "plant"}}
    leaves = [
        Sample(metadata={SESSION_ROLLOUT_METRICS_KEY: stale}),
        Sample(metadata={SESSION_ROLLOUT_METRICS_KEY: stale}),
    ]
    spec_info = {
        "spec_num_correct_drafts": 1,
        "spec_num_proposed_drafts": 2,
        "spec_verify_ct": 1,
        "completion_tokens": 2,
    }
    tracer = _Tracer(SamplesReply(samples=leaves, session_metadata=_session_metadata(spec_info), empty_reason=None))
    _patch_agent(monkeypatch, tracer)

    output = await agentic_tool_call.generate(_generate_input(sglang_speculative_algorithm="EAGLE"))

    expected = {"session_id": "sid-1", "metrics": {"spec_info": spec_info}}
    assert [sample.metadata[SESSION_ROLLOUT_METRICS_KEY] for sample in output.samples] == [expected, expected]


@pytest.mark.asyncio
async def test_v2_rejects_missing_server_session_metrics(monkeypatch):
    # [FUNCTION DOC] With a reply whose session metadata is empty (EAGLE configured), asserts
    # generate raises a key error naming the session metrics key.
    tracer = _Tracer(SamplesReply(samples=[Sample()], session_metadata={}, empty_reason=None))
    _patch_agent(monkeypatch, tracer)

    with pytest.raises(KeyError, match=SESSION_ROLLOUT_METRICS_KEY):
        await agentic_tool_call.generate(_generate_input(sglang_speculative_algorithm="EAGLE"))


@pytest.mark.asyncio
async def test_v2_rejects_metrics_from_another_session(monkeypatch):
    # [FUNCTION DOC] Makes the reply's metrics carry session id "sid-2" while the tracer's session
    # is "sid-1" and asserts generate raises a value error saying it does not match the collected
    # session.
    session_metadata = _session_metadata()
    session_metadata[SESSION_ROLLOUT_METRICS_KEY]["session_id"] = "sid-2"
    tracer = _Tracer(SamplesReply(samples=[Sample()], session_metadata=session_metadata, empty_reason=None))
    _patch_agent(monkeypatch, tracer)

    with pytest.raises(ValueError, match="does not match the collected session"):
        await agentic_tool_call.generate(_generate_input(sglang_speculative_algorithm="EAGLE"))


@pytest.mark.asyncio
async def test_v2_rejects_unavailable_metrics_from_successful_collect(monkeypatch):
    # [FUNCTION DOC] Sets the reply's metrics payload to none and asserts generate raises a value
    # error saying a successful session collect must carry metrics.
    session_metadata = _session_metadata()
    session_metadata[SESSION_ROLLOUT_METRICS_KEY]["metrics"] = None
    tracer = _Tracer(SamplesReply(samples=[Sample()], session_metadata=session_metadata, empty_reason=None))
    _patch_agent(monkeypatch, tracer)

    with pytest.raises(ValueError, match="successful session collect must carry metrics"):
        await agentic_tool_call.generate(_generate_input(sglang_speculative_algorithm="EAGLE"))


_INSTANCES_ATTR_ABSENT = object()


class TestSessionServerInstancesValidation:
    # [CLASS DOC] Tests that generate validates the configured session server instances.
    @pytest.mark.asyncio
    @pytest.mark.parametrize("instances", [_INSTANCES_ATTR_ABSENT, None, []], ids=["absent", "none", "empty"])
    async def test_empty_session_server_instances_is_rejected(self, monkeypatch, instances):
        # [FUNCTION DOC] Parametrized over the instances attribute being absent (deleted), none or
        # an empty list. Patches the tracer factory with a recorder and the agent loader, runs
        # generate, and asserts an assertion error saying it requires session server instances is
        # raised and that no tracer was created.
        """generate() raises the documented AssertionError when session_server_instances is absent, null or empty, without creating a tracer."""
        created_for: list[object] = []

        async def fake_create(args, *, evaluation=False, sampling_params=None):
            # [FUNCTION DOC] HELPER: Replacement tracer factory that records the arguments it was
            # created for and returns a tracer with an empty "no records" reply.
            created_for.append(args)
            return _Tracer(SamplesReply(samples=[], session_metadata={}, empty_reason="no_records"))

        monkeypatch.setattr(agentic_tool_call.OpenAIEndpointTracer, "create", fake_create)
        monkeypatch.setattr(agentic_tool_call, "load_function", lambda path: _fake_agent)

        generate_input = _generate_input()
        if instances is _INSTANCES_ATTR_ABSENT:
            del generate_input.args.session_server_instances
        else:
            generate_input.args.session_server_instances = instances

        with pytest.raises(AssertionError, match="requires session_server_instances"):
            await agentic_tool_call.generate(generate_input)

        assert created_for == []


@pytest.mark.asyncio
async def test_collection_error_propagates(monkeypatch):
    # [FUNCTION DOC] The tracer raises a runtime error; asserts generate propagates it unchanged
    # (unlike the timeout case, which is converted to an aborted sample).
    tracer = _Tracer(error=RuntimeError("samples unavailable"))
    _patch_agent(monkeypatch, tracer)

    with pytest.raises(RuntimeError, match="samples unavailable"):
        await agentic_tool_call.generate(_generate_input())
