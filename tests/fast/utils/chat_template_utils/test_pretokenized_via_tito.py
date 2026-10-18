# [MODULE DOC] Tests for the decode-roundtrip append-only verifiers
# verify_append_only_via_tito_instance and run_all_checks_via_tito (miles.utils.test_utils
# chat_template_verify): they must pass on registered TITO families, fail on the original unfixed
# Qwen3 chat template, and fail on a test-local Qwen3 tokenizer variant whose merge omits the
# newline fix-up. Registers as a CPU test in the stage-b-cpu suite (estimated 120 seconds). Needs
# real Hugging Face tokenizers (downloaded) and, for one test, a local DeepSeek V4 checkpoint path
# (skipped when absent).
"""Unit tests for ``verify_append_only_via_tito_instance`` /
``run_all_checks_via_tito``: PASS on registered TITO families, FAIL on the
unfixed Qwen3 chat template, FAIL on a test-local ``_BuggyQwen3TITOTokenizer``
that omits the ``\\n`` insertion at the ``<|im_end|>`` boundary.
"""

from copy import deepcopy
from pathlib import Path

import pytest
from tests.ci.ci_register import register_cpu_ci
from transformers import AutoTokenizer

register_cpu_ci(est_time=120, suite="stage-b-cpu", labels=[])


from miles.utils.chat_template_utils import TITOTokenizerType, resolve_fixed_chat_template
from miles.utils.chat_template_utils.tito_tokenizer import Qwen3TITOTokenizer
from miles.utils.test_utils.chat_template_verify import run_all_checks_via_tito, verify_append_only_via_tito_instance
from miles.utils.test_utils.mock_trajectories import SingleToolTrajectory

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _setup_tokenizer_with_registered_template(
    model_id: str,
    family: TITOTokenizerType,
):
    # [FUNCTION DOC] HELPER: Load a fresh Hugging Face tokenizer for the model id (network or
    # cache), resolve the family's fixed template and kwargs, and if a template file exists
    # overwrite the tokenizer's chat template with its text. Returns (tokenizer, copy of the extra
    # kwargs as a plain dictionary).
    """Mirror what production wiring does at startup.

    Loads tokenizer, resolves the family's ``FIXED_TEMPLATE`` (resolution is
    role-independent), and applies the fixed template (if any) onto
    ``tokenizer.chat_template``. Returns ``(tokenizer, extra_kwargs)``.

    A fresh tokenizer instance per call avoids state-mutation hazards from
    overwriting ``chat_template``.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    fixed_path, extra_kwargs = resolve_fixed_chat_template(family)
    if fixed_path is not None:
        with open(fixed_path) as f:
            tokenizer.chat_template = f.read()
    return tokenizer, dict(extra_kwargs)


# ---------------------------------------------------------------------------
# (1) PASS on registered families
# ---------------------------------------------------------------------------


_PASS_PARAMS = [
    pytest.param(TITOTokenizerType.QWEN3, "Qwen/Qwen3-0.6B", id="qwen3"),
    pytest.param(TITOTokenizerType.QWEN35, "Qwen/Qwen3.5-0.8B", id="qwen35"),
    pytest.param(TITOTokenizerType.QWEN36, "Qwen/Qwen3.6-35B-A3B", id="qwen36"),
    pytest.param(TITOTokenizerType.QWEN38_SMALL, "Qwen/Qwen3.8-27B", id="qwen38small"),
    pytest.param(TITOTokenizerType.QWEN4_EXP, "Qwen/Qwen3.8-Flash-Next", id="qwen4exp"),
    pytest.param(TITOTokenizerType.QWENNEXT, "Qwen/Qwen3-4B-Thinking-2507", id="qwennext"),
    pytest.param(TITOTokenizerType.GLM47, "zai-org/GLM-4.7-Flash", id="glm47"),
    pytest.param(TITOTokenizerType.GLM53, "zai-org/GLM-5.3", id="glm53"),
    pytest.param(TITOTokenizerType.GLM53, "zai-org/GLM-5.3-Flash", id="glm53_flash"),
]


@pytest.mark.parametrize("family,model_id", _PASS_PARAMS)
def test_via_tito_pass_on_registered_families(family, model_id):
    # [FUNCTION DOC] Parametrized over Qwen, GLM families and model ids. Sets up the tokenizer
    # with the registered template, runs all via-TITO checks (thinking "on" only for GLM 5.3, "both"
    # otherwise) with the extra template kwargs, and asserts there are no failed results; the
    # assertion message lists up to five failures with case name and error.
    """Registered TITO families round-trip cleanly via decode-roundtrip."""
    tokenizer, extra_kwargs = _setup_tokenizer_with_registered_template(model_id, family)
    results = run_all_checks_via_tito(
        tokenizer,
        family,
        thinking="on" if family == TITOTokenizerType.GLM53 else "both",
        extra_template_kwargs=extra_kwargs,
    )
    failures = [r for r in results if not r.passed]
    assert not failures, (
        f"Expected all PASS for {family.value} via TITO primitive; "
        f"got {len(failures)} FAIL(s) out of {len(results)}:\n"
        + "\n".join(f"  [{r.case_name}] {r.error}" for r in failures[:5])
    )


# ---------------------------------------------------------------------------
# (1b) PASS on DeepSeek V4 (official-encoder family, local checkpoint only)
# ---------------------------------------------------------------------------


_DEEPSEEK_V4_MODEL = "/cluster-storage/models/deepseek-ai/DeepSeek-V4-Flash"


def test_via_tito_pass_on_deepseek_v4():
    # [FUNCTION DOC] Skips unless the local DeepSeek V4 Flash checkpoint directory exists. Loads
    # its tokenizer (remote code trusted), runs all via-TITO checks for the DeepSeek V4 family in
    # both thinking modes with a copy of the family's kwargs, and asserts there are no failures
    # (message lists up to five).
    """DSv4's encoder folds contiguous tool/user turns into one ``<｜User｜>``
    block and auto-appends the assistant opener after a user tail; the
    subclass's real-history diff + opener strip must round-trip on both
    registered surfaces (the ``{tool, user}`` cuts land mid-user-block).
    """
    if not Path(_DEEPSEEK_V4_MODEL).exists():
        pytest.skip(f"DeepSeek V4 tokenizer not found: {_DEEPSEEK_V4_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(_DEEPSEEK_V4_MODEL, trust_remote_code=True)
    _fixed_path, extra_kwargs = resolve_fixed_chat_template(TITOTokenizerType.DEEPSEEKV4)
    results = run_all_checks_via_tito(
        tokenizer,
        TITOTokenizerType.DEEPSEEKV4,
        thinking="both",
        extra_template_kwargs=dict(extra_kwargs),
    )
    failures = [r for r in results if not r.passed]
    assert not failures, (
        f"Expected all PASS for deepseekv4 via TITO primitive; "
        f"got {len(failures)} FAIL(s) out of {len(results)}:\n"
        + "\n".join(f"  [{r.case_name}] {r.error}" for r in failures[:5])
    )


# ---------------------------------------------------------------------------
# (2) FAIL on the original unfixed Qwen3 chat template
# ---------------------------------------------------------------------------


def test_via_tito_fail_on_original_qwen3_template():
    # [FUNCTION DOC] Loads the Qwen3-0.6B tokenizer and keeps its original chat template, runs all
    # via-TITO checks for the Qwen3 family in both thinking modes, and asserts at least one result failed.
    """The original Qwen3 chat template uses ``loop.last`` and breaks append-only.

    Bypass ``resolve_fixed_chat_template`` entirely — keep the HF default
    ``tokenizer.chat_template`` and assert the primitive surfaces a FAIL.
    """
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    # Do NOT overwrite tokenizer.chat_template — keep the broken HF default.

    # Cast wide: thinking=both + multi-user-turn surface so trajectories that
    # actually advance ``last_query_index`` between prefix and full are exercised.
    # Those are the ones where ``loop.index0 > ns.last_query_index`` truncation
    # in the original Qwen3 template renders the same assistant turn differently
    # depending on the boundary position.
    results = run_all_checks_via_tito(
        tokenizer,
        TITOTokenizerType.QWEN3,
        thinking="both",
    )
    failures = [r for r in results if not r.passed]
    assert failures, "Expected ≥1 FAIL on the original (unfixed) Qwen3 chat template; got all PASS."


# ---------------------------------------------------------------------------
# (3) FAIL on a test-local buggy subclass
# ---------------------------------------------------------------------------


class _BuggyQwen3TITOTokenizer(Qwen3TITOTokenizer):
    # [CLASS DOC] HELPER: Test-only Qwen3 TITO tokenizer whose token merge deliberately skips the
    # newline insertion after a trailing end-of-message token, to prove the verifier detects it.
    """Test-only Qwen3 variant with the ``\\n`` boundary insertion deleted.

    Real ``Qwen3TITOTokenizer.merge_tokens`` appends ``self._newline_id`` after
    a trailing ``<|im_end|>`` because the model stops without emitting the
    newline the chat template would otherwise produce.  This variant skips
    that fixup; the decode-roundtrip primitive is expected to surface it as a
    single-character diff at the prefix-suffix junction.
    """

    def merge_tokens(self, old_messages, new_messages, pretokenized_token_ids, *, template_args=None):
        # [FUNCTION DOC] HELPER: Tokenize the additional (new minus old) messages with the parent's
        # helper and return the pretokenized ids followed by those incremental ids, with no newline
        # boundary fix-up. Returns a new list; inputs are not mutated.
        incremental = self.tokenize_additional_messages(old_messages, new_messages, template_args=template_args)
        # Intentionally omit the `+\n` insertion — that's the bug we're catching.
        return list(pretokenized_token_ids) + incremental


def test_via_tito_fail_on_buggy_qwen3_subclass():
    # [FUNCTION DOC] Builds the buggy tokenizer over a Qwen3-0.6B tokenizer with the registered
    # template and verifies the single-tool trajectory with the first 3 messages pretokenized.
    # Asserts the result did not pass and its error message contains "Decode-roundtrip mismatch".
    """A buggy ``merge_tokens`` produces a junction-level diff that the verifier surfaces."""
    tokenizer, _ = _setup_tokenizer_with_registered_template("Qwen/Qwen3-0.6B", TITOTokenizerType.QWEN3)
    buggy = _BuggyQwen3TITOTokenizer(tokenizer)

    result = verify_append_only_via_tito_instance(
        buggy,
        tokenizer,
        deepcopy(SingleToolTrajectory.MESSAGES),
        pretokenized_num_message=3,
        tools=SingleToolTrajectory.TOOLS,
        case_name="buggy_qwen3-single_tool-N3",
    )
    assert not result.passed, "Expected FAIL on _BuggyQwen3TITOTokenizer (omits the `+\\n` boundary patch); got PASS."
    assert "Decode-roundtrip mismatch" in (
        result.error or ""
    ), f"Expected decode-roundtrip diff in error message; got: {result.error}"
