# [MODULE DOC] HELPER: Verification toolkit for the append-only chat-template invariant, shared by
# the command-line verifier and tests. Layers: (1) text-level checks that a prefix render of the
# first N messages is a prefix of the full render, raising or returning result objects; (2) a
# catalogue of test cases built from mock trajectories (each trajectory expanded at each of its
# pretokenize positions) with selection by allowed append roles and thinking mode, naming helpers
# and a coverage report; (3) batch runners over the catalogue; (4) token-level verification through
# a TITO tokenizer instance, checking that merging the encoded prefix with the appendix and
# decoding reproduces the full render exactly. Pure computation apart from tokenizer and template
# calls; no I/O.
"""Verify that a chat template satisfies the append-only invariant.

The append-only invariant means: rendering the first N messages (without
generation prompt) produces a string that is an exact prefix of rendering
all messages (with generation prompt).  This is required by sglang's
pretokenized prefix mechanism for agentic workflows.

Core functions are used by both the CLI script
(``scripts/tools/verify_chat_template.py``) and the test suite
(``tests/fast/utils/chat_template_utils/test_pretokenized_chat.py``).
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from miles.utils.chat_template_utils import TITOTokenizerType
from miles.utils.chat_template_utils.template import apply_chat_template_from_str
from miles.utils.chat_template_utils.tito_tokenizer import ALL_APPEND_ROLES

if TYPE_CHECKING:
    from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer


def simulate_pretokenized_path(
    chat_template: str,
    messages: list[dict],
    pretokenized_num_message: int,
    tools: list[dict] | None = None,
    **template_kwargs,
) -> str:
    # [FUNCTION DOC] Renders the first N messages without a generation prompt and all messages
    # with a generation prompt (same tools and extra template keyword arguments), and returns the
    # full render if the first render is a string prefix of it. Raises a value error showing the
    # lengths and the last 200 characters of both sides at the divergence point otherwise.
    """Simulate the pretokenized incremental path at text level.

    1. Render first N messages (no generation prompt) -> prefix_text
    2. Render ALL messages (with generation prompt) -> full_text
    3. Verify prefix_text is a prefix of full_text

    Raises ``ValueError`` on prefix mismatch.
    """
    prefix_text = apply_chat_template_from_str(
        chat_template,
        messages[:pretokenized_num_message],
        add_generation_prompt=False,
        tools=tools,
        **template_kwargs,
    )

    full_text = apply_chat_template_from_str(
        chat_template,
        messages,
        add_generation_prompt=True,
        tools=tools,
        **template_kwargs,
    )

    if not full_text.startswith(prefix_text):
        raise ValueError(
            f"Prefix mismatch!\n"
            f"prefix_text ({len(prefix_text)} chars):\n{repr(prefix_text[-200:])}\n\n"
            f"full_text at same position:\n{repr(full_text[:len(prefix_text)][-200:])}"
        )

    return full_text


def get_standard_result(
    chat_template: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    **template_kwargs,
) -> str:
    # [FUNCTION DOC] Returns the reference rendering: the template applied to all messages with a
    # generation prompt, with the given tools and extra template keyword arguments.
    """Standard path: render all messages with generation prompt."""
    return apply_chat_template_from_str(
        chat_template,
        messages,
        add_generation_prompt=True,
        tools=tools,
        **template_kwargs,
    )


def assert_pretokenized_equals_standard(chat_template, messages, pretokenized_num_message, tools=None, **kwargs):
    # [FUNCTION DOC] Test assertion: computes the standard render and the simulated incremental
    # render and raises an assertion error if they are not equal. The incremental path itself may
    # raise a value error on a prefix mismatch. Note both calls receive the same message list
    # object (no copying), so a template that mutates messages could affect the second call.
    """Assert pretokenized incremental path produces same text as standard full render."""
    standard = get_standard_result(chat_template, messages, tools=tools, **kwargs)
    pretokenized = simulate_pretokenized_path(chat_template, messages, pretokenized_num_message, tools=tools, **kwargs)
    assert pretokenized == standard, f"Pretokenized (N={pretokenized_num_message}) != standard"


# ---------------------------------------------------------------------------
# Non-raising verification API for CLI / programmatic use
# ---------------------------------------------------------------------------


@dataclass
class VerifyResult:
    # [CLASS DOC] Outcome of one verification case: the case name, a pass flag and an optional
    # error message (absent on success).
    """Result of a single append-only verification case."""

    case_name: str
    passed: bool
    error: str | None = None


def verify_append_only(
    chat_template: str,
    messages: list[dict],
    pretokenized_num_message: int,
    tools: list[dict] | None = None,
    case_name: str = "",
    **template_kwargs,
) -> VerifyResult:
    # [FUNCTION DOC] Non-raising text-level check. Renders the standard result and the simulated
    # incremental result, each on its own deep copy of the messages (so template mutation cannot
    # leak), and returns a failed result with a fixed message when they differ. A value error
    # from the incremental path (prefix mismatch) becomes a failed result carrying that text;
    # any other exception becomes a failed result of the form exception type name, colon, message.
    # Otherwise returns a passed result.
    """Check that the template satisfies the append-only invariant.

    Returns a ``VerifyResult`` instead of raising, making it suitable for
    batch verification in CLI scripts.
    """
    try:
        standard = get_standard_result(chat_template, deepcopy(messages), tools=tools, **template_kwargs)
        pretokenized = simulate_pretokenized_path(
            chat_template, deepcopy(messages), pretokenized_num_message, tools=tools, **template_kwargs
        )
        if pretokenized != standard:
            return VerifyResult(
                case_name=case_name, passed=False, error=f"Pretokenized (N={pretokenized_num_message}) != standard"
            )
        return VerifyResult(case_name=case_name, passed=True)
    except ValueError as e:
        return VerifyResult(case_name=case_name, passed=False, error=str(e))
    except Exception as e:
        return VerifyResult(case_name=case_name, passed=False, error=f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Built-in test cases (shared between CLI and test suite)
# ---------------------------------------------------------------------------
#
# Trajectories expose two class attributes used for verify-layer filtering:
#
#   * ``APPEND_ROLES: frozenset[str]`` — non-assistant roles that appear after
#     the first assistant message.  Compared with FixedTemplate capability.
#   * ``IS_THINKING: bool`` — any assistant carries ``reasoning_content``.
#     Drives ``--thinking`` and whether ``enable_thinking`` kwarg is passed.
#
# This shared verifier intentionally covers only non-assistant append trajectories;
# dedicated family CPU tests and the session verifier cover injected assistant input.
#
# Both are declared on the trajectory class (mock_trajectories.py), alongside
# ``TOOLS`` / ``PRETOKENIZE_POSITIONS`` / ``MESSAGES``.  This file only lists
# which trajectories to exercise and expands them into concrete cases.

import re  # noqa: E402

from miles.utils.test_utils.mock_trajectories import (  # noqa: E402
    IntermediateSystemThinkingTrajectory,
    IntermediateSystemTrajectory,
    LongChainThinkingTrajectory,
    LongChainTrajectory,
    MultiRoleSequenceTrajectory,
    MultiToolSingleTurnTrajectory,
    MultiTurnNoToolThinkingTrajectory,
    MultiTurnNoToolTrajectory,
    MultiTurnThinkingTrajectory,
    MultiTurnTrajectory,
    MultiUserToolChainTrajectory,
    MultiUserTurnThinkingTrajectory,
    ParallelToolsTrajectory,
    RetrySystemTrajectory,
    SimpleNoToolTrajectory,
    SingleToolThinkingTrajectory,
    SingleToolTrajectory,
)


def _short_name(cls: type) -> str:
    # [FUNCTION DOC] HELPER: Converts a trajectory class name to a snake-case case label: removes
    # every "Trajectory" substring, inserts an underscore before each capital letter except the
    # first character, and lowercases.
    name = cls.__name__.replace("Trajectory", "")
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


# Trajectories exercised by ``run_all_checks`` / the CLI.  Must be a subset of
# the classes defined in mock_trajectories.py.  Callers (CLI, tests) pick the
# applicable subset via :func:`select_cases` based on each template's supported
# append roles and thinking mode; there is no global "exclude" list here.
_TRAJECTORIES: list[type] = [
    SingleToolTrajectory,
    MultiTurnTrajectory,
    MultiToolSingleTurnTrajectory,
    ParallelToolsTrajectory,
    LongChainTrajectory,
    MultiUserToolChainTrajectory,
    RetrySystemTrajectory,
    IntermediateSystemTrajectory,
    SimpleNoToolTrajectory,
    MultiTurnNoToolTrajectory,
    SingleToolThinkingTrajectory,
    MultiTurnThinkingTrajectory,
    LongChainThinkingTrajectory,
    MultiUserTurnThinkingTrajectory,
    IntermediateSystemThinkingTrajectory,
    MultiTurnNoToolThinkingTrajectory,
    MultiRoleSequenceTrajectory,
]


@dataclass(frozen=True)
class CaseSpec:
    # [CLASS DOC] Immutable description of one verification case: its name (short trajectory name
    # plus the pretokenize position), the trajectory class it came from, the number of leading
    # messages to pretokenize, the tool definitions (or none), the set of non-assistant roles that
    # appear after the first assistant message, and whether any assistant carries reasoning.
    """One verify case with classification metadata copied from its trajectory."""

    case_name: str
    traj_cls: type
    pretokenize_n: int
    tools: list[dict] | None
    append_roles: frozenset[str]
    is_thinking: bool


def _expand(traj_cls: type) -> list[CaseSpec]:
    # [FUNCTION DOC] HELPER: Builds one case per pretokenize position of the trajectory class,
    # named short name, "-N", position, copying the class's tools, append roles and thinking flag.
    """Expand one trajectory into one CaseSpec per PRETOKENIZE_POSITIONS value."""
    short = _short_name(traj_cls)
    return [
        CaseSpec(
            case_name=f"{short}-N{n}",
            traj_cls=traj_cls,
            pretokenize_n=n,
            tools=traj_cls.TOOLS,
            append_roles=traj_cls.APPEND_ROLES,
            is_thinking=traj_cls.IS_THINKING,
        )
        for n in traj_cls.PRETOKENIZE_POSITIONS
    ]


ALL_CASES: list[CaseSpec] = [c for t in _TRAJECTORIES for c in _expand(t)]

THINKING_MODES: tuple[str, ...] = ("off", "on", "both")


def select_cases(
    *,
    allowed_append_roles: Iterable[str],
    is_thinking: bool | None = None,
) -> list[CaseSpec]:
    # [FUNCTION DOC] Filters the module-level case catalogue, preserving its order: keeps cases
    # whose append roles are a subset of the allowed roles and, if the thinking filter is not
    # none, whose thinking flag equals it. Returns a new list.
    """Select trajectory cases by append-role surface and (optionally) thinking flag.

    A case is included iff ``case.append_roles`` is a subset of
    *allowed_append_roles*, and (when *is_thinking* is not ``None``)
    ``case.is_thinking`` matches.

    The caller is responsible for including ``"tool"`` in *allowed_append_roles*
    when the session is tool-capable; this function does not silently union it.
    """
    allowed = frozenset(allowed_append_roles)
    out: list[CaseSpec] = []
    for c in ALL_CASES:
        if not c.append_roles.issubset(allowed):
            continue
        if is_thinking is not None and c.is_thinking != is_thinking:
            continue
        out.append(c)
    return out


def enable_thinking_variants(thinking: str) -> list[dict]:
    # [FUNCTION DOC] Maps a thinking mode to the list of extra-template-argument mappings to run:
    # "off" gives one empty mapping, "on" one mapping enabling thinking, "both" two mappings
    # (enabled then disabled). Raises a value error for any other mode.
    """Return the list of ``enable_thinking`` kwarg variants to apply per case.

    * ``"off"`` → ``[{}]`` (no ``enable_thinking`` kwarg).
    * ``"on"``  → ``[{"enable_thinking": True}]``.
    * ``"both"`` → ``[{"enable_thinking": True}, {"enable_thinking": False}]``.

    Both CLI (:func:`run_all_checks`) and pytest parametrize callers use this
    to avoid drifting in how the ``enable_thinking`` knob is exercised.
    """
    if thinking == "off":
        return [{}]
    if thinking == "on":
        return [{"enable_thinking": True}]
    if thinking == "both":
        return [{"enable_thinking": True}, {"enable_thinking": False}]
    raise ValueError(f"thinking must be one of {THINKING_MODES}; got {thinking!r}")


def format_case_id(case: CaseSpec, kwargs: dict) -> str:
    # [FUNCTION DOC] Builds a label: the case name alone for empty kwargs; otherwise the case name
    # followed by dash-joined parts, one per key in sorted order, where a boolean value gives
    # key_on or key_off and any other value gives key=value.
    """Human-readable label for a ``(case, template_kwargs)`` tuple.

    Used for both CLI ``VerifyResult.case_name`` and pytest test ids so the
    same tuple is identified the same way in both surfaces.  Format:

    * empty kwargs → ``case.case_name``.
    * otherwise → ``<case.case_name>-<k1>_on/off-<k2>=val`` (keys sorted;
      bool values emit ``key_on`` / ``key_off``; other values ``key=val``).
    """
    if not kwargs:
        return case.case_name
    parts: list[str] = []
    for k, v in sorted(kwargs.items()):
        if isinstance(v, bool):
            parts.append(f"{k}_{'on' if v else 'off'}")
        else:
            parts.append(f"{k}={v}")
    return f"{case.case_name}-{'-'.join(parts)}"


@dataclass
class CoverageReport:
    # [CLASS DOC] Result of the coverage check: a mapping from (thinking flag, sorted tuple of
    # extra append roles) to the names of the cases in that bucket, and the list of keys with no
    # case.
    """Coverage of cases across ``(is_thinking, append_roles \\ {tool})``.

    ``covered`` maps each combination to the case names that fall in it;
    ``missing`` lists combinations with no case.  ``tool`` is excluded from
    the role axis because it is implicitly always allowed.
    """

    covered: dict[tuple[bool, tuple[str, ...]], list[str]]
    missing: list[tuple[bool, tuple[str, ...]]]


def check_coverage(
    cases: list[CaseSpec] | None = None,
    *,
    role_universe: set[str] | None = None,
) -> CoverageReport:
    # [FUNCTION DOC] Defaults: the full case catalogue and role universe of user and system.
    # Builds every subset of the sorted role universe (as sorted tuples) crossed with thinking
    # false/true, as empty buckets; places each case in the bucket keyed by its thinking flag and
    # its append roles minus the tool role (sorted tuple), ignoring cases whose key is not a
    # bucket (roles outside the universe); returns the buckets and the keys of empty buckets.
    """Enumerate ``thinking × append-role-subset`` combinations and report gaps.

    Used as a sanity check that every meaningful append-role and thinking
    combination is backed by at least one trajectory.
    """
    if cases is None:
        cases = ALL_CASES
    if role_universe is None:
        role_universe = {"user", "system"}

    from itertools import chain, combinations

    ordered_universe = sorted(role_universe)
    all_subsets: list[tuple[str, ...]] = [
        tuple(sub)
        for sub in chain.from_iterable(combinations(ordered_universe, r) for r in range(len(ordered_universe) + 1))
    ]

    covered: dict[tuple[bool, tuple[str, ...]], list[str]] = {
        (is_thinking, sub): [] for is_thinking in (False, True) for sub in all_subsets
    }
    for c in cases:
        roles_key = tuple(sorted(c.append_roles - {"tool"}))
        key = (c.is_thinking, roles_key)
        if key in covered:
            covered[key].append(c.case_name)

    missing = [k for k, v in covered.items() if not v]
    return CoverageReport(covered=covered, missing=missing)


def run_all_checks(
    chat_template: str,
    *,
    allowed_append_roles: set[str] | frozenset[str] | None = None,
    thinking: str = "off",
    extra_template_kwargs: dict | None = None,
) -> list[VerifyResult]:
    # [FUNCTION DOC] Runs the text-level check over the filtered catalogue. Defaults the allowed
    # roles to the maximal set; raises a value error for an unknown thinking mode. Thinking
    # filter: off selects non-thinking cases, on thinking cases, both all. For every selected
    # case and every thinking variant, merges the variant with the extra kwargs (extra wins on
    # key conflict) and appends the non-raising check result run on a deep copy of the trajectory
    # messages, named by the case id. Returns the results in case then variant order.
    """Run verification cases filtered by *allowed_append_roles* and *thinking*.

    ``allowed_append_roles`` is the role surface the template may append after
    an assistant turn; defaults to the maximal four-role surface.
    Trajectories whose ``append_roles`` are not a subset are skipped.  Caller
    must include ``"tool"`` explicitly when relevant — there is no implicit
    union.

    ``thinking`` selects which ``enable_thinking`` variants are exercised —
    see :func:`enable_thinking_variants`.  When ``"both"``, **every** selected
    trajectory (thinking or not) is rerun with ``enable_thinking=True`` and
    ``enable_thinking=False``, so templates that branch on the flag are
    validated against non-reasoning input too.

    ``extra_template_kwargs`` is merged into every invocation — use it to
    thread template-specific kwargs (e.g. GLM's ``clear_thinking=False``)
    through the CLI.
    """
    if allowed_append_roles is None:
        allowed_append_roles = ALL_APPEND_ROLES
    if thinking not in THINKING_MODES:
        raise ValueError(f"thinking must be one of {THINKING_MODES}; got {thinking!r}")
    extra = extra_template_kwargs or {}

    is_thinking_filter = {"off": False, "on": True, "both": None}[thinking]
    selected = select_cases(allowed_append_roles=allowed_append_roles, is_thinking=is_thinking_filter)
    variants = enable_thinking_variants(thinking)

    results: list[VerifyResult] = []
    for case in selected:
        for variant in variants:
            kwargs = {**variant, **extra}
            results.append(
                verify_append_only(
                    chat_template,
                    deepcopy(case.traj_cls.MESSAGES),
                    case.pretokenize_n,
                    tools=case.tools,
                    case_name=format_case_id(case, kwargs),
                    **kwargs,
                )
            )

    return results


# ---------------------------------------------------------------------------
# TITO-instance verification: decode-roundtrip equality
# ---------------------------------------------------------------------------
#
# The string-based primitive above asserts text-prefix at the chat-template
# layer.  This is necessary but not sufficient for production correctness —
# production runs ``get_tito_tokenizer(...)`` and exercises ``merge_tokens``
# (model-specific token-level boundary patches) plus
# ``tokenize_additional_messages`` (renders the complete appendix under a
# synthetic ``[_DUMMY_SYSTEM, dummy_assistant]`` context, not the real history).
#
# The primitive below mirrors the production path: it instantiates the actual
# TITO subclass + HF tokenizer, runs ``merge_tokens`` against the encoded
# prefix, decodes, and asserts text equality with the canonical full render.


def verify_append_only_via_tito_instance(
    tito: TITOTokenizer,
    tokenizer: Any,
    messages: list[dict],
    pretokenized_num_message: int,
    tools: list[dict] | None = None,
    case_name: str = "",
    **template_kwargs,
) -> VerifyResult:
    # [FUNCTION DOC] Token-level check with an existing TITO tokenizer and an underlying
    # tokenizer. Never raises: any exception becomes a failed result of type name and message.
    # Steps: find the end M of the maximal run of non-assistant messages starting at the
    # pretokenize position N; fail with an explanatory error if that run is empty (message N is
    # an assistant). Deep-copy the first N messages (prefix) and the first M messages (full).
    # Obtain the TITO's default template arguments for the tools; render the prefix without and
    # the full list with a generation prompt using the TITO. Encode the prefix text without
    # special tokens and strip trailing token ids that the TITO declares as trailing (simulating
    # where the model stops). Merge tokens using the TITO with prefix, full and stripped ids,
    # decode, and compare with the full render: pass if equal, else fail with the index of the
    # first differing character and about 60 characters of context before and after, expected vs
    # actual.
    # KNOWN ISSUE: the extra template keyword arguments of this function are accepted but never
    # used here; the TITO instance must already carry them.
    """Decode-roundtrip verify with a pre-built TITO instance.

    Asserts ``decode(tito.merge_tokens(prefix_msgs, full_msgs, encode(prefix_text)))
    == full_text`` where ``prefix_text`` and ``full_text`` come from running the
    chat template through ``tokenizer`` with the same kwargs ``tito`` was built
    with.  The test-only path (e.g. ``BuggyQwen3TITOTokenizer``) uses this
    instance form directly; production-shape callers go through
    :func:`verify_append_only_via_tito`.
    """
    try:
        # This shared verifier checks the maximal non-assistant run after boundary N.
        # Production also accepts injected assistant input; dedicated family CPU
        # tests and the session verifier cover that surface.
        n = pretokenized_num_message
        m = n
        while m < len(messages) and messages[m].get("role") != "assistant":
            m += 1
        if m == n:
            return VerifyResult(
                case_name=case_name,
                passed=False,
                error=(
                    f"Empty appendix at N={n}: messages[{n}] is assistant. "
                    "PRETOKENIZE_POSITIONS must land at a post-assistant boundary "
                    "where messages[N:] starts with a non-assistant turn."
                ),
            )

        prefix_msgs = deepcopy(messages[:n])
        full_msgs = deepcopy(messages[:m])

        template_args = tito.default_template_args(tools)
        prefix_text = tito.apply_chat_template(prefix_msgs, add_generation_prompt=False, template_args=template_args)
        full_text = tito.apply_chat_template(full_msgs, add_generation_prompt=True, template_args=template_args)

        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        # Simulate production's model-stop: in production, ``pretokenized_token_ids``
        # ends where the model actually stopped — typically before the trailing
        # tokens the chat template would otherwise emit (Qwen's ``\n`` after
        # ``<|im_end|>``, GLM's ambiguous ``<|user|>``/``<|observation|>`` boundary).
        # The TITO subclass declares those as ``trailing_token_ids``.  Trim them
        # here so ``merge_tokens``'s boundary patches see the prefix in its
        # production shape so the verifier sees the same prefix the
        # subclass merge_tokens / trailing trim path operates on.
        trailing = tito.trailing_token_ids
        while prefix_ids and prefix_ids[-1] in trailing:
            prefix_ids = prefix_ids[:-1]
        merged_ids = tito.merge_tokens(prefix_msgs, full_msgs, prefix_ids, template_args=template_args)
        merged_text = tokenizer.decode(merged_ids)

        if merged_text == full_text:
            return VerifyResult(case_name=case_name, passed=True)

        # Find first divergence and quote ~60 chars of context on each side.
        common_len = min(len(merged_text), len(full_text))
        diff_idx = next(
            (i for i in range(common_len) if merged_text[i] != full_text[i]),
            common_len,
        )
        ctx_start = max(0, diff_idx - 60)
        ctx_end = diff_idx + 60
        return VerifyResult(
            case_name=case_name,
            passed=False,
            error=(
                f"Decode-roundtrip mismatch (N={pretokenized_num_message}) at char {diff_idx}\n"
                f"  expected: ...{full_text[ctx_start:ctx_end]!r}...\n"
                f"  actual:   ...{merged_text[ctx_start:ctx_end]!r}..."
            ),
        )
    except Exception as e:
        return VerifyResult(case_name=case_name, passed=False, error=f"{type(e).__name__}: {e}")


def verify_append_only_via_tito(
    tokenizer: Any,
    tito_model: TITOTokenizerType | str,
    messages: list[dict],
    pretokenized_num_message: int,
    tools: list[dict] | None = None,
    case_name: str = "",
    **template_kwargs,
) -> VerifyResult:
    # [FUNCTION DOC] Builds a TITO tokenizer for the registered model family via the factory,
    # passing a copy of the extra template kwargs as chat template kwargs (local import), then
    # delegates to the instance-based check with the same messages, position, tools, case name
    # and kwargs, returning its result.
    """Decode-roundtrip verify, building TITO from the registered family.

    Matches the production wiring at ``miles/rollout/session/sessions.py:35`` —
    the same ``get_tito_tokenizer`` factory call, with ``chat_template_kwargs``
    threaded through so ``merge_tokens`` and the dummy-context appendix render
    use the same kwargs as the reference full render.
    """
    from miles.utils.chat_template_utils import get_tito_tokenizer

    tito = get_tito_tokenizer(
        tokenizer,
        tokenizer_type=tito_model,
        chat_template_kwargs=dict(template_kwargs),
    )
    return verify_append_only_via_tito_instance(
        tito,
        tokenizer,
        messages,
        pretokenized_num_message,
        tools=tools,
        case_name=case_name,
        **template_kwargs,
    )


def run_all_checks_via_tito(
    tokenizer: Any,
    tito_model: TITOTokenizerType | str,
    *,
    thinking: str = "off",
    extra_template_kwargs: dict | None = None,
) -> list[VerifyResult]:
    # [FUNCTION DOC] Token-level batch runner. Raises a value error for an unknown thinking mode.
    # Coerces a string model name to the TITO type enum, looks up the tokenizer class's fixed
    # template and selects cases by its allowed append roles and the thinking filter. Skips cases
    # whose pretokenize position is at or past the end of the messages or lands on an assistant
    # message (empty appendix). For each remaining case and thinking variant, merges variant and
    # extra kwargs (extra wins), builds a fresh TITO and appends the roundtrip result for a deep
    # copy of the messages. Returns the results in order.
    """Same shape as :func:`run_all_checks` but routes through TITO + tokenizer.

    Per-case TITO rebuild: each (case, ``enable_thinking`` variant) gets a fresh
    TITO instance constructed with the merged kwargs, so the dummy-context
    appendix render inside ``tokenize_additional_messages`` sees the same
    ``enable_thinking`` value as the reference render.  Construction is
    millisecond-level and runs ~50 times per CLI invocation; cheap.

    The caller is responsible for setting ``tokenizer.chat_template`` (e.g. via
    ``resolve_fixed_chat_template`` lookup or ``--template`` override) before
    calling this — this function does not consult ``FIXED_TEMPLATE``.
    """
    if thinking not in THINKING_MODES:
        raise ValueError(f"thinking must be one of {THINKING_MODES}; got {thinking!r}")
    extra = extra_template_kwargs or {}

    if isinstance(tito_model, str):
        tito_model = TITOTokenizerType(tito_model)
    fixed_template = TITOTokenizerType.get_tokenizer_class(tito_model).FIXED_TEMPLATE
    is_thinking_filter = {"off": False, "on": True, "both": None}[thinking]
    selected = select_cases(
        allowed_append_roles=fixed_template.allowed_append_roles,
        is_thinking=is_thinking_filter,
    )
    variants = enable_thinking_variants(thinking)

    results: list[VerifyResult] = []
    for case in selected:
        # This shared verifier requires a non-empty non-assistant appendix. Cases
        # ending at an assistant remain covered at the text-prefix layer; dedicated
        # family CPU tests and the session verifier cover injected assistant input.
        msgs = case.traj_cls.MESSAGES
        n = case.pretokenize_n
        if n >= len(msgs) or msgs[n].get("role") == "assistant":
            continue
        for variant in variants:
            kwargs = {**variant, **extra}
            results.append(
                verify_append_only_via_tito(
                    tokenizer,
                    tito_model,
                    deepcopy(case.traj_cls.MESSAGES),
                    case.pretokenize_n,
                    tools=case.tools,
                    case_name=format_case_id(case, kwargs),
                    **kwargs,
                )
            )

    return results
