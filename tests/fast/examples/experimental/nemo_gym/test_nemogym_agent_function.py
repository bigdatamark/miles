# [MODULE DOC]
# Offline unit tests for the NeMo Gym agent function module and its data conversion script (both
# imported by bare name). The agent function posts a run request to a NeMo Gym server whose url
# comes from an environment variable; the posting function is replaced by fakes that record the
# url and payload or fail. Tests cover the request contract (instance fields at the top level, a
# policy base url ending in /v1 derived from the session url, sampling parameters mapped onto
# response creation parameters, no legacy fields), the reward and eval report mapping, none on
# transport failure or timeout, and conversion of a raw instance file into the training format.
"""Offline unit tests for the NeMo Gym adapter (no network, no GPU).

Runs on every PR (stage-a-cpu, by the tests/fast/ convention); locally:

    pytest tests/fast/examples/experimental/nemo_gym -q

Covers the /run request contract the mini_swe_agent_2 server expects (instance
fields at the top level, policy_base_url override, sampling mapped onto
responses_create_params), the response mapping, and the failure semantics
(transport failure -> None so the sample keeps its recorded session).
"""

import asyncio
import json

import download_and_process_data
import nemogym_agent_function as naf


def run_async(coro):
    # [FUNCTION DOC] HELPER: Runs the coroutine to completion in a fresh event loop and returns its result.
    return asyncio.run(coro)


_INSTANCE_METADATA = {
    "instance_id": "django__django-10973",
    "repo": "django/django",
    "base_commit": "ddb2936",
    "problem_statement": "Use subprocess.run ...",
    "subset": "gym",
    "split": "train",
}


def _capture_post(captured, response=None):
    # [FUNCTION DOC] HELPER: Returns an async fake for the json posting function. It stores the
    # url and payload into the given mapping and returns the given response, or by default a reward
    # of 0.0 with empty metadata.
    async def fake_post(url, payload, **kwargs):
        # [FUNCTION DOC] HELPER: Records the url and payload; returns the canned response (extra keyword arguments ignored).
        captured["url"] = url
        captured["payload"] = payload
        return response if response is not None else {"reward": 0.0, "metadata": {}}

    return fake_post


# --- request contract -----------------------------------------------------


def test_run_body_carries_instance_fields_and_policy_override(monkeypatch):
    # [FUNCTION DOC] Sets the gym url environment variable and runs the agent with a session url,
    # sampling kwargs (temperature 0.8, top_p 0.9, max_tokens 4096) and instance metadata.
    # Asserts the post went to the gym url plus /run; every metadata field is at the top level of
    # the body; the policy base url is the session url plus /v1; the response creation params are
    # an empty input list plus temperature, top_p and max output tokens (max_tokens renamed); and
    # the legacy sglang url and sampling params fields are absent.
    captured = {}
    monkeypatch.setattr(naf, "post_json", _capture_post(captured))
    monkeypatch.setenv("NEMO_GYM_URL", "http://gym:12000")

    run_async(
        naf.run(
            base_url="http://trainer:30000/sessions/abc",
            prompt="ignored",
            request_kwargs={"temperature": 0.8, "top_p": 0.9, "max_tokens": 4096},
            metadata=dict(_INSTANCE_METADATA),
        )
    )

    assert captured["url"] == "http://gym:12000/run"
    body = captured["payload"]
    # Instance fields at the top level: the server uses body.model_dump() as
    # the instance dict (image selection, eval).
    for key, value in _INSTANCE_METADATA.items():
        assert body[key] == value
    assert body["policy_base_url"] == "http://trainer:30000/sessions/abc/v1"
    assert body["responses_create_params"] == {
        "input": [],
        "temperature": 0.8,
        "top_p": 0.9,
        "max_output_tokens": 4096,
    }
    # The fork-era channels must be gone.
    assert "sglang_url" not in body
    assert "sampling_params" not in body


def test_sampling_params_omitted_when_unset(monkeypatch):
    # [FUNCTION DOC] With no request kwargs, asserts the response creation params are just an empty input list.
    captured = {}
    monkeypatch.setattr(naf, "post_json", _capture_post(captured))

    run_async(naf.run(base_url="http://t:1/sessions/s", prompt="", request_kwargs={}, metadata={}))

    assert captured["payload"]["responses_create_params"] == {"input": []}


# --- response mapping -----------------------------------------------------


def test_reward_and_eval_report_mapping(monkeypatch):
    # [FUNCTION DOC] The fake server answers reward 1.0 with an eval report as metadata; asserts
    # the agent returns a mapping with that reward and the report under the eval report key.
    eval_report = {"django__django-10973": {"patch_successfully_applied": True}}
    captured = {}
    monkeypatch.setattr(naf, "post_json", _capture_post(captured, {"reward": 1.0, "metadata": eval_report}))

    result = run_async(naf.run(base_url="http://t:1/sessions/s", prompt="", metadata={}))

    assert result == {"reward": 1.0, "eval_report": eval_report}


def test_transport_failure_returns_none(monkeypatch):
    # [FUNCTION DOC] The fake post raises a runtime error (connection refused); asserts the agent returns none.
    async def failing_post(url, payload, **kwargs):
        # [FUNCTION DOC] HELPER: Always raises a runtime error.
        raise RuntimeError("connection refused")

    monkeypatch.setattr(naf, "post_json", failing_post)

    assert run_async(naf.run(base_url="http://t:1/sessions/s", prompt="", metadata={})) is None


def test_timeout_returns_none(monkeypatch):
    # [FUNCTION DOC] The fake post sleeps ten seconds while the run timeout environment variable
    # is 0.01 seconds; asserts the agent returns none.
    async def slow_post(url, payload, **kwargs):
        # [FUNCTION DOC] HELPER: Sleeps ten seconds and returns nothing.
        await asyncio.sleep(10)

    monkeypatch.setattr(naf, "post_json", slow_post)
    monkeypatch.setenv("NEMO_GYM_RUN_TIMEOUT", "0.01")

    assert run_async(naf.run(base_url="http://t:1/sessions/s", prompt="", metadata={})) is None


# --- data conversion ------------------------------------------------------


def test_convert_to_miles_format(tmp_path):
    # [FUNCTION DOC] Writes one raw instance as a json line to a temp file, converts it to the
    # training format with split train, and reads the output line. Asserts the prompt is the
    # problem statement and the metadata keeps the instance id and patch and adds subset gym and
    # split train. Side effect: files in the temp path.
    src = tmp_path / "raw.jsonl"
    instance = {"instance_id": "x__y-1", "repo": "x/y", "problem_statement": "fix it", "patch": "diff"}
    src.write_text(json.dumps(instance) + "\n")
    dst = tmp_path / "miles.jsonl"

    download_and_process_data.convert_to_miles_format(str(src), str(dst), split="train")

    row = json.loads(dst.read_text())
    assert row["prompt"] == "fix it"
    # Full instance preserved in metadata, plus the subset/split the server's
    # image selection and eval need.
    assert row["metadata"]["instance_id"] == "x__y-1"
    assert row["metadata"]["patch"] == "diff"
    assert row["metadata"]["subset"] == "gym"
    assert row["metadata"]["split"] == "train"
