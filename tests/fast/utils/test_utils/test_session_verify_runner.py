# [MODULE DOC] Tests for the session verification runner and its callers: the training argument
# string built from a namespace (flag presence/omission rules per option), the defaults and action
# table of the verification script, the end-to-end helper functions that run one model or both
# session server versions (batch size alignment, Anthropic pass configuration, thresholds), the
# metrics file assertions (append tool coverage, hard mismatches, special token count threshold,
# verifier errors, soft assistant text ratio), preservation of the metrics sidecar on failure, and
# the environment passed to the training job per wire format. Subprocess and cluster actions are
# replaced with fakes; nothing is launched.
import argparse
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from scripts.tools.verify_session_tito_tokenizer import _print_action_table, _with_session_verify_defaults
from tests.e2e.sglang.test_session_server_multi_role import _common

from miles.utils.arguments import parse_args_train_backend
from miles.utils.test_utils import session_verify_runner
from miles.utils.test_utils.session_verify_runner import (
    SESSION_VERIFY_INVARIANT_ARGS,
    assert_session_verify_metrics,
    namespace_to_train_args,
)
from miles.utils.tracking_utils.ci_history import RECORD_DIR_ENV


def test_session_verify_defaults_select_fsdp_during_backend_preparse(monkeypatch):
    # [FUNCTION DOC] Applies the script's default arguments to a minimal argument list, sets the
    # process argument vector to it (reverted at teardown), and asserts the backend pre-parse
    # selects the fsdp backend.
    argv = _with_session_verify_defaults(["--rollout-batch-size", "1"])
    monkeypatch.setattr(sys, "argv", ["verify_session_tito_tokenizer.py", *argv])

    assert parse_args_train_backend() == "fsdp"


def _build_args(**overrides) -> str:
    # [FUNCTION DOC] HELPER: Builds a namespace from the invariant arguments plus a typical model
    # setup (checkpoint, qwen3 tito model, 2 GPUs per engine, one actor node of 8 GPUs, 4 samples
    # per prompt, 3 cycles, rollback failure mode, parsers, and unset optional sglang settings),
    # applies the overrides on top, and returns the training argument string from the converter.
    values = {
        **SESSION_VERIFY_INVARIANT_ARGS,
        "hf_checkpoint": "/root/models/test-model",
        "tito_model": "qwen3",
        "rollout_num_gpus_per_engine": 2,
        "actor_num_nodes": 1,
        "actor_num_gpus_per_node": 8,
        "n_samples_per_prompt": 4,
        "session_verify_cycles": 3,
        "tool_call_failure_mode": "rollback",
        "sglang_reasoning_parser": "qwen3",
        "sglang_tool_call_parser": "qwen25",
        "sglang_context_length": None,
        "sglang_kv_cache_dtype": None,
        "sglang_mamba_full_memory_ratio": None,
        "sglang_cuda_graph_backend_prefill": None,
        "anthropic_intermediate_system_expectation": None,
    }
    values.update(overrides)
    return namespace_to_train_args(argparse.Namespace(**values))


def test_namespace_to_train_args_uses_default_rollout_max_response_len():
    # [FUNCTION DOC] The default rollout max response length flag is 8192.
    train_args = _build_args()

    assert "--rollout-max-response-len 8192" in train_args


def test_namespace_to_train_args_allows_model_specific_rollout_max_response_len():
    # [FUNCTION DOC] An override of 16384 appears in the flag.
    train_args = _build_args(rollout_max_response_len=16384)

    assert "--rollout-max-response-len 16384" in train_args


def test_namespace_to_train_args_keeps_ci_test_enabled_for_fsdp_debug_rollout():
    # [FUNCTION DOC] The arguments include the fsdp train backend flag and the CI test flag.
    train_args = _build_args()

    assert "--train-backend fsdp" in train_args
    assert "--ci-test" in train_args


def test_namespace_to_train_args_preserves_resolved_rollout_gpu_count():
    # [FUNCTION DOC] A resolved rollout GPU count of 8 without colocation appears as the flag.
    train_args = _build_args(rollout_num_gpus=8, colocate=False)

    assert "--rollout-num-gpus 8" in train_args


def test_namespace_to_train_args_defaults_to_session_server_v2():
    # [FUNCTION DOC] The session server flag defaults to version v2.
    train_args = _build_args()

    assert "--use-session-server v2" in train_args


def test_namespace_to_train_args_allows_session_server_v1():
    # [FUNCTION DOC] An override to v1 appears in the flag.
    train_args = _build_args(use_session_server="v1")

    assert "--use-session-server v1" in train_args


def test_namespace_to_train_args_emits_session_message_matcher():
    # [FUNCTION DOC] The matcher flag defaults to strict and can be overridden to loose tool call.
    assert "--session-message-matcher strict" in _build_args()
    assert "--session-message-matcher loose_tool_call" in _build_args(session_message_matcher="loose_tool_call")


def test_namespace_to_train_args_emits_anthropic_intermediate_system_expectation():
    # [FUNCTION DOC] The expectation flag is absent when unset and present with value required
    # when set.
    assert "--anthropic-intermediate-system-expectation" not in _build_args()
    assert "--anthropic-intermediate-system-expectation required" in _build_args(
        anthropic_intermediate_system_expectation="required"
    )


def test_namespace_to_train_args_emits_special_token_count_threshold():
    # [FUNCTION DOC] The special token count threshold flag is omitted when unset or zero, and
    # emitted with the value when positive (0.05).
    assert "--ci-tito-special-token-count-threshold" not in _build_args()
    assert "--ci-tito-special-token-count-threshold" not in _build_args(ci_tito_special_token_count_threshold=0.0)
    assert "--ci-tito-special-token-count-threshold 0.05" in _build_args(ci_tito_special_token_count_threshold=0.05)


def test_namespace_to_train_args_has_no_append_role_policy_flag():
    # [FUNCTION DOC] The arguments never mention an allowed append roles flag.
    train_args = _build_args()

    assert "allowed-append-roles" not in train_args


def test_namespace_to_train_args_omits_context_length_by_default():
    # [FUNCTION DOC] The sglang context length flag is absent when unset.
    train_args = _build_args()

    assert "--sglang-context-length" not in train_args


def test_namespace_to_train_args_emits_model_context_length():
    # [FUNCTION DOC] A context length of 32768 appears as the flag.
    train_args = _build_args(sglang_context_length=32768)

    assert "--sglang-context-length 32768" in train_args


def test_namespace_to_train_args_omits_model_mamba_cache_config_by_default():
    # [FUNCTION DOC] The kv cache dtype and mamba full memory ratio flags are absent when unset.
    train_args = _build_args()

    assert "--sglang-kv-cache-dtype" not in train_args
    assert "--sglang-mamba-full-memory-ratio" not in train_args


def test_namespace_to_train_args_emits_model_mamba_cache_config():
    # [FUNCTION DOC] A kv cache dtype fp8_e4m3 and a ratio 4.59 appear as flags with values.
    train_args = _build_args(sglang_kv_cache_dtype="fp8_e4m3", sglang_mamba_full_memory_ratio=4.59)

    assert "--sglang-kv-cache-dtype fp8_e4m3" in train_args
    assert "--sglang-mamba-full-memory-ratio 4.59" in train_args


def test_namespace_to_train_args_emits_cuda_graph_max_batch_size():
    # [FUNCTION DOC] A decode cuda graph max batch size of 8 appears as a flag.
    train_args = _build_args(sglang_cuda_graph_max_bs_decode=8)

    assert "--sglang-cuda-graph-max-bs-decode 8" in train_args


def test_action_table_uses_requested_cycle_count(capsys):
    # [FUNCTION DOC] Prints the action table for four roles and 1 cycle, captures standard
    # output, and asserts exactly two tool result lines and no ninth numbered row.
    _print_action_table(["assistant", "system", "tool", "user"], cycles=1)

    output = capsys.readouterr().out
    assert output.count(". tool_result") == 2
    assert "9. " not in output


def test_namespace_to_train_args_omits_prefill_cuda_graph_backend_by_default():
    # [FUNCTION DOC] The prefill cuda graph backend flag is absent when unset.
    train_args = _build_args()

    assert "--sglang-cuda-graph-backend-prefill" not in train_args


def test_namespace_to_train_args_emits_prefill_cuda_graph_backend():
    # [FUNCTION DOC] A prefill backend of disabled appears as a flag with that value.
    train_args = _build_args(sglang_cuda_graph_backend_prefill="disabled")

    assert "--sglang-cuda-graph-backend-prefill disabled" in train_args


@pytest.mark.parametrize(("n_samples_per_prompt", "expected_global_batch_size"), [(1, 16), (4, 64)])
def test_run_one_aligns_global_batch_size_with_sample_count(
    monkeypatch, n_samples_per_prompt, expected_global_batch_size
):
    # [FUNCTION DOC] Parametrized over 1 and 4 samples per prompt (global batch 16 and 64).
    # Replaces the session verify runner in the shared e2e module with a capture function, builds
    # a model config, runs one model, and asserts the captured args: global batch size as
    # expected, rollout batch size 16, max response length 4096, kv cache dtype, mamba ratio and
    # prefill backend forwarded, and the wire format openai.
    captured = {}
    monkeypatch.setattr(
        _common,
        "run_session_verify",
        lambda args, *, wire_format: captured.update(args=args, wire_format=wire_format),
    )
    config = _common.ModelConfig(
        model_name="test-model",
        reasoning_parser="qwen3",
        tool_call_parser="qwen25",
        tito_model="qwen3",
        n_samples_per_prompt=n_samples_per_prompt,
        rollout_max_response_len=4096,
        kv_cache_dtype="fp8_e4m3",
        mamba_full_memory_ratio=4.59,
        cuda_graph_backend_prefill="disabled",
    )

    _common.run_one(config)

    assert captured["args"].global_batch_size == expected_global_batch_size
    assert captured["args"].rollout_batch_size == 16
    assert captured["args"].rollout_max_response_len == 4096
    assert captured["args"].sglang_kv_cache_dtype == "fp8_e4m3"
    assert captured["args"].sglang_mamba_full_memory_ratio == 4.59
    assert captured["args"].sglang_cuda_graph_backend_prefill == "disabled"
    assert captured["wire_format"] == "openai"


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_run_one_uses_requested_session_server_version(monkeypatch, version):
    # [FUNCTION DOC] Parametrized over v1 and v2; asserts the captured args carry the requested
    # session server version and the wire format is openai.
    captured = {}
    monkeypatch.setattr(
        _common,
        "run_session_verify",
        lambda args, *, wire_format: captured.update(args=args, wire_format=wire_format),
    )
    config = _common.ModelConfig(
        model_name="test-model",
        reasoning_parser="qwen3",
        tool_call_parser="qwen25",
        tito_model="qwen3",
    )

    _common.run_one(config, session_server_version=version)

    assert captured["args"].use_session_server == version
    assert captured["wire_format"] == "openai"


def test_run_one_rejects_anthropic_on_v1():
    # [FUNCTION DOC] Requesting the anthropic endpoint on session server v1 raises a value error
    # saying it requires v2.
    config = _common.ModelConfig(
        model_name="test-model",
        reasoning_parser="qwen3",
        tool_call_parser="qwen25",
        tito_model="qwen3",
    )

    with pytest.raises(ValueError, match="requires session server v2"):
        _common.run_one(config, session_server_version="v1", endpoint="anthropic")


def test_run_one_requires_explicit_anthropic_intermediate_system_expectation():
    # [FUNCTION DOC] The anthropic endpoint with a config lacking the intermediate system
    # expectation raises a value error saying the expectation is required.
    config = _common.ModelConfig(
        model_name="test-model",
        reasoning_parser="qwen3",
        tool_call_parser="qwen25",
        tito_model="qwen3",
    )

    with pytest.raises(ValueError, match="requires an intermediate-system expectation"):
        _common.run_one(config, endpoint="anthropic")


@pytest.mark.parametrize(("n_samples_per_prompt", "expected_global_batch_size"), [(1, 8), (4, 32)])
@pytest.mark.parametrize(
    ("threshold_overrides", "expected_thresholds"),
    [
        ({"assistant_text_threshold": 0.25}, [0.25, 0.25, 0.25]),
        (
            {"assistant_text_threshold": 0.25, "anthropic_assistant_text_threshold": 1.0},
            [0.25, 0.25, 1.0],
        ),
    ],
)
def test_run_both_versions_adds_v2_anthropic_pass(
    monkeypatch,
    n_samples_per_prompt,
    expected_global_batch_size,
    threshold_overrides,
    expected_thresholds,
):
    # [FUNCTION DOC] Parametrized over 1 or 4 samples per prompt and two threshold override sets
    # (one shared assistant text threshold 0.25, or that plus a separate anthropic threshold 1.0).
    # The runner is replaced by a capture list. Running both versions must produce three passes:
    # openai v1, openai v2 and anthropic v2. Asserts wire formats, session server versions, rollout
    # batch size 8 each, global batch size by sample count, assistant text thresholds as
    # expected, special token count thresholds all 0.0, matchers strict, strict and loose tool
    # call, the anthropic expectation only on the third pass (required), and the custom generate
    # and agent function paths (invariant defaults for the first two, the anthropic ones for the
    # third).
    captured = []
    monkeypatch.setattr(
        _common,
        "run_session_verify",
        lambda args, *, wire_format: captured.append((args, wire_format)),
    )
    config = _common.ModelConfig(
        model_name="test-model",
        reasoning_parser="qwen3",
        tool_call_parser="qwen25",
        tito_model="qwen3",
        n_samples_per_prompt=n_samples_per_prompt,
        anthropic_intermediate_system_expectation="required",
        **threshold_overrides,
    )

    _common.run_both_versions(config)

    args = [item[0] for item in captured]
    assert [item[1] for item in captured] == ["openai", "openai", "anthropic"]
    assert [item.use_session_server for item in args] == ["v1", "v2", "v2"]
    assert [item.rollout_batch_size for item in args] == [8, 8, 8]
    assert [item.global_batch_size for item in args] == [expected_global_batch_size] * 3
    assert [item.assistant_text_threshold for item in args] == expected_thresholds
    assert [item.ci_tito_special_token_count_threshold for item in args] == [0.0, 0.0, 0.0]
    assert [item.session_message_matcher for item in args] == ["strict", "strict", "loose_tool_call"]
    assert [item.anthropic_intermediate_system_expectation for item in args] == [None, None, "required"]
    assert [item.custom_generate_function_path for item in args] == [
        SESSION_VERIFY_INVARIANT_ARGS["custom_generate_function_path"],
        SESSION_VERIFY_INVARIANT_ARGS["custom_generate_function_path"],
        _common._ANTHROPIC_GENERATE,
    ]
    assert [item.custom_agent_function_path for item in args] == [
        SESSION_VERIFY_INVARIANT_ARGS["custom_agent_function_path"],
        SESSION_VERIFY_INVARIANT_ARGS["custom_agent_function_path"],
        _common._ANTHROPIC_AGENT,
    ]


def test_run_both_versions_can_disable_anthropic(monkeypatch):
    # [FUNCTION DOC] With the anthropic verification disabled in the config, only two openai
    # passes (v1 then v2) are captured.
    captured = []
    monkeypatch.setattr(
        _common,
        "run_session_verify",
        lambda args, *, wire_format: captured.append((args, wire_format)),
    )
    config = _common.ModelConfig(
        model_name="test-model",
        reasoning_parser="minimax-append-think",
        tool_call_parser="minimax-m2",
        tito_model="minimax_m27",
        verify_anthropic=False,
    )

    _common.run_both_versions(config)

    assert [wire_format for _, wire_format in captured] == ["openai", "openai"]
    assert [args.use_session_server for args, _ in captured] == ["v1", "v2"]


def test_namespace_to_train_args_omits_expert_parallel_for_single_expert():
    # [FUNCTION DOC] The expert parallel size flag is absent when unset.
    train_args = _build_args()

    assert "--sglang-expert-parallel-size" not in train_args


def test_namespace_to_train_args_emits_expert_parallel_for_moe():
    # [FUNCTION DOC] An expert parallel size of 8 appears as a flag.
    train_args = _build_args(sglang_ep_size=8)

    assert "--sglang-expert-parallel-size 8" in train_args


def test_namespace_to_train_args_omits_speculative_decoding_by_default():
    # [FUNCTION DOC] No speculative decoding flags appear by default.
    train_args = _build_args()

    assert "--sglang-speculative-" not in train_args


def test_namespace_to_train_args_enables_eagle_speculative_decoding():
    # [FUNCTION DOC] Enabling speculation emits the EAGLE algorithm, 2 steps, top-k 1 and 3 draft
    # tokens flags.
    train_args = _build_args(enable_spec=True)

    assert "--sglang-speculative-algorithm EAGLE" in train_args
    assert "--sglang-speculative-num-steps 2" in train_args
    assert "--sglang-speculative-eagle-topk 1" in train_args
    assert "--sglang-speculative-num-draft-tokens 3" in train_args


def _write_metrics(path, entries: list[dict]) -> None:
    # [FUNCTION DOC] HELPER: Writes each entry as one JSON line (with a trailing newline) to the
    # path, overwriting it.
    path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n")


def test_session_verify_metrics_accepts_cross_sample_append_tool(tmp_path):
    # [FUNCTION DOC] Two samples, only one with an append tool event: the metrics assertion
    # passes (the requirement is across samples).
    metrics_path = tmp_path / "metrics.jsonl"
    _write_metrics(
        metrics_path,
        [
            {"driver_events": ["initial", "append_user"], "had_assistant_mismatch": False},
            {"driver_events": ["initial", "append_tool"], "had_assistant_mismatch": False},
        ],
    )

    assert_session_verify_metrics(str(metrics_path), assistant_text_threshold=0.1)


def test_session_verify_metrics_requires_at_least_one_append_tool(tmp_path):
    # [FUNCTION DOC] No sample has an append tool event: assertion error saying no sample
    # produced one.
    metrics_path = tmp_path / "metrics.jsonl"
    _write_metrics(metrics_path, [{"driver_events": ["initial", "append_user"], "had_assistant_mismatch": False}])

    with pytest.raises(AssertionError, match="no sample produced an append_tool action"):
        assert_session_verify_metrics(str(metrics_path), assistant_text_threshold=0.1)


def test_session_verify_metrics_can_skip_multi_role_append_tool_gate(tmp_path):
    # [FUNCTION DOC] With the append tool requirement turned off, a sample with only an
    # anthropic tool use event passes.
    metrics_path = tmp_path / "metrics.jsonl"
    _write_metrics(
        metrics_path,
        [{"driver_events": ["anthropic_tool_use"], "had_assistant_mismatch": False}],
    )

    assert_session_verify_metrics(str(metrics_path), assistant_text_threshold=0.1, require_append_tool=False)


def test_session_verify_metrics_hard_mismatch_precedes_soft_threshold(tmp_path):
    # [FUNCTION DOC] A sample with a hard mismatch (special token count) and threshold 0 raises
    # an assertion error about hard mismatches naming that type, even though it has no append
    # tool event.
    metrics_path = tmp_path / "metrics.jsonl"
    _write_metrics(
        metrics_path,
        [
            {
                "driver_events": [],
                "had_assistant_mismatch": True,
                "hard_mismatch_count": 1,
                "hard_mismatch_types": ["special_token_count"],
                "hard_mismatch_example": {"type": "special_token_count"},
            }
        ],
    )

    with pytest.raises(AssertionError, match="hard TITO mismatches.*special_token_count"):
        assert_session_verify_metrics(str(metrics_path), assistant_text_threshold=0.0)


@pytest.mark.parametrize(
    ("hard_types", "threshold", "raises"),
    [
        (["special_token_count"], 0.5, False),
        (["special_token_count"], 0.25, True),
        (["special_token_count", "special_token_type"], 0.5, True),
        (["non_assistant_text"], 0.5, True),
    ],
)
def test_session_verify_metrics_special_token_count_threshold(tmp_path, hard_types, threshold, raises):
    # [FUNCTION DOC] Two samples, one carrying the given hard mismatch types and one clean, with
    # the special token count threshold given. Parametrized: one count mismatch at 0.5 passes
    # (1 of 2 samples within the ratio); at 0.25 raises; a count plus a type mismatch raises; a
    # non assistant text mismatch raises. Raising means an assertion error about hard mismatches.
    metrics_path = tmp_path / "metrics.jsonl"
    _write_metrics(
        metrics_path,
        [
            {
                "driver_events": ["append_tool"],
                "had_assistant_mismatch": False,
                "hard_mismatch_count": len(hard_types),
                "hard_mismatch_types": hard_types,
                "hard_mismatch_example": {"type": hard_types[0]},
            },
            {"driver_events": ["append_tool"], "had_assistant_mismatch": False},
        ],
    )

    def check():
        # [FUNCTION DOC] HELPER: Runs the metrics assertion with assistant text threshold 1.0 and
        # the parametrized special token count threshold.
        assert_session_verify_metrics(
            str(metrics_path), assistant_text_threshold=1.0, special_token_count_threshold=threshold
        )

    if raises:
        with pytest.raises(AssertionError, match="hard TITO mismatches"):
            check()
    else:
        check()


@pytest.mark.parametrize(
    ("include_clean_sample", "stage", "message", "completed_samples"),
    [
        (True, "anthropic_session_verify_agent.generate", "expected six requests", 1),
        (False, "session_verify_agent.run_agent", "missing rollback", 0),
    ],
)
def test_session_verify_metrics_rejects_verifier_error(
    tmp_path, include_clean_sample, stage, message, completed_samples
):
    # [FUNCTION DOC] Parametrized over an anthropic generate stage error with one clean sample,
    # and a run agent stage error with none. The file holds a verification error record (stage,
    # type, message) and optionally a clean sample. Asserts an assertion error whose text shows
    # the completed sample count and the verifier's message.
    metrics_path = tmp_path / "metrics.jsonl"
    entries = [{"verification_error": {"stage": stage, "type": "AssertionError", "message": message}}]
    if include_clean_sample:
        entries.append({"driver_events": ["append_tool"], "had_assistant_mismatch": False})
    _write_metrics(metrics_path, entries)

    with pytest.raises(AssertionError, match=f"completed_samples={completed_samples}.*{message}"):
        assert_session_verify_metrics(str(metrics_path), assistant_text_threshold=1.0)


def test_session_verify_metrics_keeps_assistant_text_soft(tmp_path):
    # [FUNCTION DOC] A sample with an assistant text mismatch passes at threshold 1.0 and raises
    # an assertion error about the assistant text mismatch ratio at threshold 0.0.
    metrics_path = tmp_path / "metrics.jsonl"
    _write_metrics(
        metrics_path,
        [{"driver_events": ["append_tool"], "had_assistant_mismatch": True}],
    )

    assert_session_verify_metrics(str(metrics_path), assistant_text_threshold=1.0)
    with pytest.raises(AssertionError, match="assistant_text mismatch ratio"):
        assert_session_verify_metrics(str(metrics_path), assistant_text_threshold=0.0)


@pytest.mark.parametrize("failure_phase", ["execute_train", "post_gate"])
def test_run_session_verify_preserves_sidecar_on_failure(monkeypatch, tmp_path, failure_phase):
    # [FUNCTION DOC] Parametrized over failure in the train launch (a subprocess error) or in the
    # post-run metrics gate (an assertion error). Replaces the temporary file creator (returning
    # an open descriptor and a path in the temporary directory), the parser resolver, data
    # preparation, proxy clearing, model download and argument conversion, and the default
    # backend with one whose train launch writes a hard mismatch metrics line to the sidecar and
    # optionally fails. Runs the verification and asserts the expected error type, that the
    # original sidecar path no longer exists, and that a renamed copy with suffix failed holds
    # the written line.
    """A failed session verification preserves metrics beside the temporary sidecar."""
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_fd = os.open(metrics_path, os.O_CREAT | os.O_RDWR)
    monkeypatch.setattr(session_verify_runner.tempfile, "mkstemp", lambda **kwargs: (metrics_fd, str(metrics_path)))
    monkeypatch.setattr(
        session_verify_runner,
        "resolve_reasoning_and_tool_call_parser",
        lambda tito_model, reasoning_parser, tool_call_parser: (reasoning_parser, tool_call_parser),
    )
    monkeypatch.setattr(session_verify_runner, "_ensure_prompt_data", lambda: None)
    monkeypatch.setattr(session_verify_runner, "_clear_proxy_env", lambda: None)
    monkeypatch.setattr(session_verify_runner, "_ensure_model_downloaded", lambda checkpoint, *, backend: checkpoint)
    monkeypatch.setattr(session_verify_runner, "namespace_to_train_args", lambda args: "train args")

    def fake_execute_train(**kwargs):
        # [FUNCTION DOC] HELPER: Writes one metrics line with hard mismatch count 1 to the sidecar
        # path from the extra environment variables, then raises a called process error when the
        # failure phase is the train launch.
        sidecar = kwargs["extra_env_vars"]["MILES_SESSION_VERIFY_METRICS_PATH"]
        with open(sidecar, "w") as f:
            f.write('{"hard_mismatch_count": 1}\n')
        if failure_phase == "execute_train":
            raise subprocess.CalledProcessError(1, "ray job submit")

    backend = SimpleNamespace(execute_train=fake_execute_train)
    monkeypatch.setattr(
        session_verify_runner.command_utils,
        "default_config",
        lambda: SimpleNamespace(create_backend=lambda: backend),
    )
    args = argparse.Namespace(
        tito_model="qwen3",
        sglang_reasoning_parser="qwen3",
        sglang_tool_call_parser="qwen25",
        hf_checkpoint="/models/test",
        actor_num_gpus_per_node=8,
        assistant_text_threshold=0.2,
    )

    expected_error = subprocess.CalledProcessError if failure_phase == "execute_train" else AssertionError
    with pytest.raises(expected_error):
        session_verify_runner.run_session_verify(args)

    assert not metrics_path.exists()
    assert (tmp_path / "metrics.jsonl.failed").read_text() == '{"hard_mismatch_count": 1}\n'


@pytest.mark.parametrize(("wire_format", "disables_history"), [("openai", False), ("anthropic", True)])
def test_session_verify_env_isolates_anthropic_from_openai_history(wire_format, disables_history):
    # [FUNCTION DOC] Parametrized over openai (history recording untouched) and anthropic (history
    # record directory variable present and empty). Asserts the environment has the tito model and
    # metrics path, and the record directory variable is present exactly for anthropic.
    args = argparse.Namespace(tito_model="qwen3")

    env = session_verify_runner._session_verify_env(args, "/tmp/metrics.jsonl", wire_format=wire_format)

    assert env["MILES_TITO_MODEL"] == "qwen3"
    assert env["MILES_SESSION_VERIFY_METRICS_PATH"] == "/tmp/metrics.jsonl"
    assert (RECORD_DIR_ENV in env) is disables_history
    if disables_history:
        assert env[RECORD_DIR_ENV] == ""
