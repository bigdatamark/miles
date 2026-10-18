# [MODULE DOC]
# Offline tests for three parts of the metric-history gate. (1) Step selection over a metric time
# series of step/value pairs: the "last", "all" and explicit step-list modes, skipping of
# non-numeric points, non-finite handling, empty/duplicate/null-step errors. (2) Constraint
# evaluation: a two-sided corridor around a reference whose width is the larger of a relative
# part and an absolute floor on each side, scaled from the reference only. (3) Parsing of
# gate declarations from test source files through the syntax tree: defaults, the table of
# standard-metric defaults, literal-only arguments, canonical (sorted, whitespace-free) JSON
# declaration keys, and loud rejection of unknown, removed, malformed or missing arguments.
# Declaration fixtures are written to a temporary directory. Registers itself as a CPU test in
# the stage-a-cpu suite.
"""Offline unit tests for step selection, constraints, and register_ci_gate
parsing (including the canonical declaration keys)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from tests.ci.ci_register import CIRegistry, HWBackend, register_cpu_ci, ut_parse_one_file
from tests.ci.metric_history.constraints import evaluate_constraint
from tests.ci.metric_history.register import GATE_DEFAULTS, parse_ci_gate_specs
from tests.ci.metric_history.selection import SelectionError, select

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

# --- step selection -----------------------------------------------------------


def test_last_picks_last_numeric_point():
    # [FUNCTION DOC] Selecting "last" from a three-point series yields exactly one extraction whose
    # identity step is the reduction sentinel (minus one), whose landing step is the final point's
    # step (2) and whose value is that point's value.
    got = select([[0, 0.1], [1, 0.2], [2, 0.35]], "last")
    assert len(got) == 1
    # Identity step is the -1 reduction sentinel; the landing step is reporting.
    assert got[0].step == -1
    assert got[0].at_step == 2
    assert got[0].value == pytest.approx(0.35)


def test_selection_skips_bool_and_none_values():
    # [FUNCTION DOC] A series whose early points hold a boolean and a missing value and whose last
    # point is a real number: selecting "last" ignores the non-numeric points and returns the
    # numeric value (booleans are not treated as numbers).
    # bool sneaks through isinstance(int); None is not a number. Non-numeric
    # points are ignored -- only numeric points count toward selection.
    series = [[0, True], [1, None], [2, 2.5]]
    got = select(series, "last")
    assert got[0].value == pytest.approx(2.5)


def test_last_non_finite_at_selected_point_errors():
    # [FUNCTION DOC] For a final point that is nan or infinity, selecting "last" raises a
    # selection error whose message mentions "non-finite".
    for bad in (float("nan"), float("inf")):
        with pytest.raises(SelectionError, match="non-finite"):
            select([[0, 1.0], [1, bad]], "last")


def test_last_ignores_non_selected_non_finite():
    # [FUNCTION DOC] A nan at an earlier point does not matter when "last" is selected; the finite
    # final point (step 1, value 2.5) is returned.
    # A mid-series NaN is not the selected coordinate: last gates the actual
    # last point, which is finite here.
    got = select([[0, float("nan")], [1, 2.5]], "last")
    assert got[0].at_step == 1
    assert got[0].value == pytest.approx(2.5)


def test_all_non_finite_errors():
    # [FUNCTION DOC] Selecting "all" over a series containing nan at step 1 raises a selection
    # error with the message "all: non-finite value nan at step 1".
    with pytest.raises(SelectionError, match="all: non-finite value nan at step 1"):
        select([[0, 1.0], [1, float("nan")]], "all")


def test_steps_non_finite_at_named_step_errors():
    # [FUNCTION DOC] Selecting the explicit step list containing step 1 when that step holds
    # negative infinity raises a selection error saying "non-finite value -inf at required step 1".
    with pytest.raises(SelectionError, match="non-finite value -inf at required step 1"):
        select([[0, 1.0], [1, float("-inf")]], [1])


def test_empty_series_errors_clearly():
    # [FUNCTION DOC] An empty series raises a selection error for each mode: "last", "all" and an
    # explicit step list.
    for steps in ("last", "all", [0]):
        with pytest.raises(SelectionError):
            select([], steps)


def test_all_fans_out_one_extraction_per_step():
    # [FUNCTION DOC] Selecting "all" over a three-point series returns one extraction per point,
    # in order, each with identity step equal to landing step and its own value.
    got = select([[0, 1.0], [1, 2.0], [2, 3.0]], "all")
    assert [(e.step, e.at_step, e.value) for e in got] == [
        (0, 0, 1.0),
        (1, 1, 2.0),
        (2, 2, 3.0),
    ]


def test_all_null_step_errors():
    # [FUNCTION DOC] Selecting "all" over a series with a missing step index raises a selection
    # error mentioning "no step index".
    with pytest.raises(SelectionError, match="no step index"):
        select([[0, 1.0], [None, 2.0]], "all")


def test_all_duplicate_step_errors():
    # [FUNCTION DOC] Selecting "all" over a series where step 0 appears twice raises a selection
    # error mentioning "duplicate step".
    with pytest.raises(SelectionError, match="duplicate step"):
        select([[0, 1.0], [0, 2.0]], "all")


def test_steps_picks_named_steps():
    # [FUNCTION DOC] Selecting the explicit steps 0 and 2 from a three-point series returns just
    # those two extractions with their step and value.
    got = select([[0, 0.001], [1, 0.5], [2, 0.9]], [0, 2])
    assert [(e.step, e.value) for e in got] == [(0, 0.001), (2, 0.9)]


def test_steps_missing_named_step_errors():
    # [FUNCTION DOC] Requesting step 3 from a series that only has steps 0 and 1 raises a selection
    # error saying "required step 3 missing".
    with pytest.raises(SelectionError, match="required step 3 missing"):
        select([[0, 0.001], [1, 0.5]], [3])


# --- constraints --------------------------------------------------------------


def test_symmetric_corridor():
    # [FUNCTION DOC] With a twenty percent relative band on both sides and no absolute floor
    # around a reference of 1.0: a current value of 1.1 passes; 1.3 and 0.7 fail.
    c = {"rel_up": 0.20, "abs_floor_up": 0.0, "rel_down": 0.20, "abs_floor_down": 0.0}
    assert evaluate_constraint(c, 1.1, 1.0).ok
    assert not evaluate_constraint(c, 1.3, 1.0).ok
    assert not evaluate_constraint(c, 0.7, 1.0).ok


def test_band_scales_from_ref_only():
    # [FUNCTION DOC] With a fifty percent band around a reference of 0.5, the upper edge is 0.75:
    # 0.74 passes while 0.76 and 0.99 fail, proving the band is derived from the reference only
    # and a deviating current value cannot widen its own band.
    # The tolerance is fixed by history: a deviating cur must not widen its
    # own band (a max over both magnitudes would pass 0.99 here, an effective
    # ceiling of 2x ref instead of the declared 1.5x).
    c = {"rel_up": 0.50, "abs_floor_up": 0.0, "rel_down": 0.50, "abs_floor_down": 0.0}
    assert evaluate_constraint(c, 0.74, 0.5).ok
    assert not evaluate_constraint(c, 0.76, 0.5).ok
    assert not evaluate_constraint(c, 0.99, 0.5).ok


def test_abs_floor_covers_near_zero():
    # [FUNCTION DOC] With a zero reference the relative band vanishes but an absolute floor of one
    # millionth remains: 1e-7 passes and 0.5 fails.
    # A reference near zero makes the relative band vanish; the floor remains.
    c = {"rel_up": 0.20, "abs_floor_up": 1e-6, "rel_down": 0.20, "abs_floor_down": 1e-6}
    assert evaluate_constraint(c, 1e-7, 0.0).ok
    assert not evaluate_constraint(c, 0.5, 0.0).ok


def test_each_side_band_is_max_of_rel_and_floor():
    # [FUNCTION DOC] Around reference 0.5 with up band (30 percent, floor 0.1) and down band
    # (10 percent, floor 0.1): the upper limit is 0.65 (relative wins) and the lower limit is 0.4
    # (floor wins), i.e. each side's width is the larger of relative part and floor.
    c = {"rel_up": 0.30, "abs_floor_up": 0.1, "rel_down": 0.10, "abs_floor_down": 0.1}
    out = evaluate_constraint(c, 0.5, 0.5)
    assert out.hi == pytest.approx(0.65)  # rel side wins: 0.3*0.5 > 0.1
    assert out.lo == pytest.approx(0.4)  # floor wins: 0.1*0.5 < 0.1


def test_asymmetric_sides():
    # [FUNCTION DOC] With a tight upper band (10 percent) and a loose lower band (80 percent)
    # around reference 2.0: 2.1 and 0.5 pass; 2.3 and 0.3 fail.
    # Tight upper band, loose-but-finite lower band: a moderate drop passes,
    # a rise beyond band and a collapse far below both fail.
    c = {"rel_up": 0.10, "abs_floor_up": 0.0, "rel_down": 0.80, "abs_floor_down": 0.0}
    assert evaluate_constraint(c, 2.1, 2.0).ok
    assert not evaluate_constraint(c, 2.3, 2.0).ok
    assert evaluate_constraint(c, 0.5, 2.0).ok
    assert not evaluate_constraint(c, 0.3, 2.0).ok


# --- register_ci_gate parsing -----------------------------------------------


def _make_fixture(body: str, tmp_path: Path, name: str = "test_gatefix.py") -> str:
    # [FUNCTION DOC] HELPER: Write a fixture test file into the temporary directory under the given
    # name (default "test_gatefix.py"). The body text is dedented and leading newlines are removed
    # before writing. Returns the file path as a string.
    p = tmp_path / name
    p.write_text(textwrap.dedent(body).lstrip("\n"))
    return str(p)


def test_parse_single_spec_with_defaults(tmp_path):
    # [FUNCTION DOC] Parses a fixture with one gate declaration (steps "all", relative up/down 0.2
    # only). Asserts one spec: metric key and steps as written, constraint normalized with the
    # missing absolute floors filled as 0.0, enforcement off, no allowlist reason, and the
    # filename equal to the fixture path.
    path = _make_fixture(
        """
        from tests.ci.ci_register import register_cuda_ci
        from tests.ci.metric_history import register_ci_gate
        register_cuda_ci(est_time=600, suite="stage-c-8-gpu-h100")
        register_ci_gate(
            metric_key="train/grad_norm",
            steps="all",
            constraint={"rel_up": 0.20, "rel_down": 0.20},
        )
        """,
        tmp_path,
    )
    specs = parse_ci_gate_specs(path)
    assert len(specs) == 1
    s = specs[0]
    assert s.metric_key == "train/grad_norm"
    assert s.steps == "all"
    assert s.constraint == {"rel_up": 0.20, "abs_floor_up": 0.0, "rel_down": 0.20, "abs_floor_down": 0.0}
    assert s.enforce is False
    assert s.allowlist_reason is None
    assert s.filename == path


def test_parse_all_fields(tmp_path):
    # [FUNCTION DOC] Parses a declaration with explicit step list, partial constraint (missing
    # lower floor), enforcement on and an allowlist reason; asserts each is carried to the spec and
    # the missing floor defaults to 0.0.
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(
            metric_key="train/ppo_kl",
            steps=[0, 1],
            constraint={"abs_floor_up": 1e-6, "rel_up": 0.5, "rel_down": 0.8},
            enforce=True,
            allowlist_reason="known noisy",
        )
        """,
        tmp_path,
    )
    s = parse_ci_gate_specs(path)[0]
    assert s.steps == [0, 1]
    assert s.constraint == {"rel_up": 0.5, "abs_floor_up": 1e-6, "rel_down": 0.8, "abs_floor_down": 0.0}
    assert s.enforce is True
    assert s.allowlist_reason == "known noisy"


def test_declaration_keys_are_canonical_json(tmp_path):
    # [FUNCTION DOC] Two declarations written with irregular spacing and unsorted dictionary keys.
    # Asserts the derived steps key and constraint key are canonical JSON: sorted keys, no
    # whitespace (the list becomes zero-one without spaces; the string "last" stays quoted).
    # Dict keys are sorted and every literal is whitespace-free regardless of
    # how the author wrote it; this pins the exact stored-identity format.
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/ppo_kl",
                         steps=[0,  1],
                         constraint={"abs_floor_up": 0.02, "abs_floor_down": 0.02})
        register_ci_gate(metric_key="train/x", steps="last",
                         constraint={"rel_up": 0.2, "rel_down": 0.2})
        """,
        tmp_path,
    )
    fanned, reduced = parse_ci_gate_specs(path)
    assert fanned.steps_key == "[0,1]"
    assert fanned.constraint_key == '{"abs_floor_down":0.02,"abs_floor_up":0.02}'
    assert reduced.steps_key == '"last"'


def test_declaration_keys_use_raw_literal_not_normalized(tmp_path):
    # [FUNCTION DOC] Parses a declaration with only relative band params. Asserts the normalized
    # constraint has the floors filled in, but the constraint key is derived from the literal as
    # written (no absolute-floor entries), so changing code defaults can never rewrite stored keys.
    # The normalized constraint fills the omitted params from code; the key
    # must come from the literal as written, so a code-side default change
    # can never silently rewrite keys and reset every series.
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/x",
                         steps="last", constraint={"rel_up": 0.2, "rel_down": 0.2})
        """,
        tmp_path,
    )
    s = parse_ci_gate_specs(path)[0]
    assert s.constraint == {"rel_up": 0.2, "abs_floor_up": 0.0, "rel_down": 0.2, "abs_floor_down": 0.0}
    assert s.constraint_key == '{"rel_down":0.2,"rel_up":0.2}'
    assert "abs_floor" not in s.constraint_key


def test_abs_optional_rel_defaults_to_zero(tmp_path):
    # [FUNCTION DOC] A constraint giving only absolute floors parses, with both relative parts
    # defaulting to 0.0.
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(
            metric_key="train/ppo_kl",
            steps="last",
            constraint={"abs_floor_up": 1e-6, "abs_floor_down": 1e-6},
        )
        """,
        tmp_path,
    )
    s = parse_ci_gate_specs(path)[0]
    assert s.constraint == {"rel_up": 0.0, "abs_floor_up": 1e-6, "rel_down": 0.0, "abs_floor_down": 1e-6}


def test_parse_multiple_specs(tmp_path):
    # [FUNCTION DOC] A fixture with two declarations yields two specs, in source order.
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/grad_norm",
                         steps="all", constraint={"rel_up": 0.2, "rel_down": 0.2})
        register_ci_gate(metric_key="rollout/raw_reward",
                         steps="last",
                         constraint={"rel_up": 0.5, "rel_down": 0.2})
        """,
        tmp_path,
    )
    specs = parse_ci_gate_specs(path)
    assert [s.metric_key for s in specs] == ["train/grad_norm", "rollout/raw_reward"]


def test_unknown_kwarg_rejected(tmp_path):
    # [FUNCTION DOC] A declaration with an unrecognized keyword argument makes the parser raise a
    # value error "unknown argument 'bogus'".
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/x",
                         steps="last", constraint={"rel_up": 0.2, "rel_down": 0.2}, bogus=3)
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="unknown argument 'bogus'"):
        parse_ci_gate_specs(path)


def test_non_literal_arg_rejected(tmp_path):
    # [FUNCTION DOC] A declaration whose steps argument is a variable name rather than a literal
    # makes the parser raise a value error saying it must be a literal.
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        X = "last"
        register_ci_gate(metric_key="train/x", steps=X,
                         constraint={"rel_up": 0.2, "rel_down": 0.2})
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="must be a literal"):
        parse_ci_gate_specs(path)


def test_non_literal_inside_dict_rejected(tmp_path):
    # [FUNCTION DOC] A constraint dictionary containing a variable as a value is also rejected with
    # a value error saying it must be a literal.
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        X = 0.2
        register_ci_gate(metric_key="train/x",
                         steps="last", constraint={"rel_up": X, "rel_down": 0.2})
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="must be a literal"):
        parse_ci_gate_specs(path)


@pytest.mark.parametrize("missing", ["metric_key", "steps", "constraint"])
def test_missing_required_field_rejected(tmp_path, missing):
    # [FUNCTION DOC] Parametrized over metric key, steps and constraint. Builds a declaration
    # omitting that one field (with the other two present, and no table default applicable for a
    # missing metric key) and asserts a value error "<field> is required".
    fields = {
        "metric_key": 'metric_key="train/x"',
        "steps": 'steps="last"',
        "constraint": 'constraint={"rel_up": 0.2, "rel_down": 0.2}',
    }
    del fields[missing]
    call = f"register_ci_gate({', '.join(fields.values())})"
    path = _make_fixture(
        f"""
        from tests.ci.metric_history import register_ci_gate
        {call}
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match=f"{missing} is required"):
        parse_ci_gate_specs(path)


def test_one_liner_fills_from_gate_defaults(tmp_path):
    # [FUNCTION DOC] A declaration with only the metric key of a standard metric (policy KL) is
    # complete: steps "all" and the full four-parameter constraint come from the defaults table;
    # the steps key and constraint key are the canonical JSON of the table literals.
    # A bare metric_key declaration on a standard metric is complete: steps and
    # constraint come from the table, and the declaration keys derive from the
    # TABLE literals (canonical JSON), not from any normalized form.
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/ppo_kl")
        """,
        tmp_path,
    )
    s = parse_ci_gate_specs(path)[0]
    assert s.steps == "all"
    assert s.constraint == {"rel_up": 0.5, "abs_floor_up": 0.02, "rel_down": 0.8, "abs_floor_down": 0.02}
    assert s.steps_key == '"all"'
    assert s.constraint_key == '{"abs_floor_down":0.02,"abs_floor_up":0.02,"rel_down":0.8,"rel_up":0.5}'


def test_partial_default_written_literal_wins(tmp_path):
    # [FUNCTION DOC] A declaration giving a metric key and a constraint but no steps: steps is
    # filled from the table ("all") while the constraint key reflects the written literal rather
    # than the table's constraint.
    # Each omitted field fills independently: an explicit constraint keeps its
    # own literal (and key) while steps still comes from the table.
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="rollout/raw_reward",
                         constraint={"rel_up": 0.1, "rel_down": 0.1})
        """,
        tmp_path,
    )
    s = parse_ci_gate_specs(path)[0]
    assert s.steps == "all"  # from GATE_DEFAULTS
    assert s.constraint_key == '{"rel_down":0.1,"rel_up":0.1}'  # written literal, not the table's


def test_one_liner_without_table_entry_rejected(tmp_path):
    # [FUNCTION DOC] A bare metric key with no entry in the defaults table makes the parser raise a
    # value error "has no GATE_DEFAULTS entry".
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/not_a_standard_metric")
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="has no GATE_DEFAULTS entry"):
        parse_ci_gate_specs(path)


def test_gate_defaults_within_capture_whitelist_and_valid(tmp_path):
    # [FUNCTION DOC] Checks two defaults-table invariants. Every defaulted metric key must be in
    # the set of metric keys the tracking utility captures (otherwise the verdict would always be
    # an error). And a fixture file with one bare declaration per table key must parse into exactly
    # as many specs as table entries, so every entry passes schema validation. The fixture body is
    # built by concatenation to avoid breaking dedent.
    # Two table invariants: every defaulted key must be captured (a default for
    # an uncaptured metric guarantees an ERROR verdict), and every table entry
    # must survive the parser's own schema validation.
    from miles.utils.tracking_utils.ci_history import TARGET_METRIC_KEYS

    assert set(GATE_DEFAULTS) <= set(TARGET_METRIC_KEYS)

    # Concatenate instead of interpolating into an indented f-string: the
    # joined lines would defeat the fixture's dedent.
    body = "from tests.ci.metric_history import register_ci_gate\n" + "\n".join(
        f'register_ci_gate(metric_key="{key}")' for key in GATE_DEFAULTS
    )
    path = _make_fixture(body, tmp_path)
    specs = parse_ci_gate_specs(path)
    assert len(specs) == len(GATE_DEFAULTS)


@pytest.mark.parametrize(
    "metric_key",
    [
        "train/grad_norm",
        "train/ppo_kl",
        "train/train_rollout_logprob_abs_diff",
        "train/train_rollout_kl",
        "rollout/raw_reward",
    ],
)
def test_standard_rl_gate_defaults_capture_all_steps(metric_key):
    # [FUNCTION DOC] Parametrized over five standard RL metrics (gradient norm, policy KL,
    # train/rollout logprob abs diff, train/rollout KL, raw reward): the default steps mode in the
    # defaults table is "all".
    assert GATE_DEFAULTS[metric_key]["steps"] == "all"


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_session_tito_gate_default_is_loose(version):
    # [FUNCTION DOC] Parametrized over versions v1 and v2. The defaults entry for the session
    # token-in-token-out mismatch rate of assistant text equals exactly: steps "last" and a
    # constraint with relative up 1.0, absolute floor up 0.1, relative down 1.0 (no lower floor).
    metric_key = f"rollout/tito_session_mismatch_rate/{version}/assistant_text"
    assert GATE_DEFAULTS[metric_key] == {
        "steps": "last",
        "constraint": {"rel_up": 1.0, "abs_floor_up": 0.1, "rel_down": 1.0},
    }


def test_normal_session_e2e_declares_both_tito_history_gates():
    # [FUNCTION DOC] Finds every test file in the multi-role session-server end-to-end directory
    # (relative to this file's grandparent) and asserts the directory is non-empty and that each
    # file declares gates for both the v1 and v2 assistant-text mismatch-rate metrics.
    case_dir = Path(__file__).parents[2] / "e2e/sglang/test_session_server_multi_role"
    required = {
        "rollout/tito_session_mismatch_rate/v1/assistant_text",
        "rollout/tito_session_mismatch_rate/v2/assistant_text",
    }
    case_files = sorted(case_dir.glob("test_*.py"))
    assert case_files
    for case_file in case_files:
        declared = {spec.metric_key for spec in parse_ci_gate_specs(str(case_file))}
        assert required <= declared, case_file


def test_one_liner_runtime_is_noop():
    # [FUNCTION DOC] Calling the gate registration function at runtime with only a metric key
    # succeeds and returns nothing; validity is decided by the parser, not the signature.
    # The Python signature must accept the one-liner form at import time; the
    # parser, not the signature, decides validity.
    from tests.ci.metric_history import register_ci_gate

    assert register_ci_gate(metric_key="rollout/raw_reward") is None


def test_positional_arg_rejected(tmp_path):
    # [FUNCTION DOC] A declaration using positional arguments makes the parser raise a value error
    # "only keyword arguments".
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate("train/x", 1.0)
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="only keyword arguments"):
        parse_ci_gate_specs(path)


def test_unknown_steps_keyword_rejected(tmp_path):
    # [FUNCTION DOC] A steps string other than the allowed keywords (here a mean-of-last-N style
    # name) makes the parser raise a value error beginning 'steps must be "last", "all"'.
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/x",
                         steps="mean_last_9000", constraint={"rel_up": 0.2, "rel_down": 0.2})
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match='steps must be "last", "all"'):
        parse_ci_gate_specs(path)


def test_constraint_name_key_is_gone(tmp_path):
    # [FUNCTION DOC] A constraint still carrying the removed "name" key makes the parser raise a
    # value error "unknown key 'name' for constraint" instead of silently dropping it.
    # The old name-keyed constraint registry was merged into one band family;
    # a declaration still passing "name" must fail loud, not silently drop.
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/x",
                         steps="last", constraint={"name": "band", "rel_up": 0.2, "rel_down": 0.2})
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="unknown key 'name' for constraint"):
        parse_ci_gate_specs(path)


@pytest.mark.parametrize(
    "literal,match",
    [
        ('{"rel_down": 0.2}', "upper band param"),
        ('{"rel_up": 0.2}', "lower band param"),
    ],
    ids=["missing-up", "missing-down"],
)
def test_constraint_missing_side_rejected(tmp_path, literal, match):
    # [FUNCTION DOC] Parametrized over a constraint missing the upward side and one missing the
    # downward side. The fixture embeds the literal; the parser must raise a value error matching
    # "upper band param" or "lower band param" respectively, since a side with no written band
    # param would have band zero.
    # An all-default side has band 0 and fails on any deviation; the parser
    # demands at least one written band param per side.
    path = _make_fixture(
        f"""
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/x",
                         steps="last", constraint={literal})
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match=match):
        parse_ci_gate_specs(path)


def test_non_string_non_list_steps_rejected(tmp_path):
    # [FUNCTION DOC] A steps value that is an integer (neither keyword string nor list) is rejected
    # with the value error beginning 'steps must be "last", "all"'.
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/x",
                         steps=0, constraint={"rel_up": 0.2, "rel_down": 0.2})
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match='steps must be "last", "all"'):
        parse_ci_gate_specs(path)


@pytest.mark.parametrize(
    "steps_literal",
    ["[]", "[1.5]", "[True]", "[-1]", "[0, 0]"],
    ids=["empty", "float", "bool", "negative", "duplicate"],
)
def test_bad_steps_list_rejected(tmp_path, steps_literal):
    # [FUNCTION DOC] Parametrized over invalid step lists: empty, a float entry, a boolean entry,
    # a negative entry, and a duplicate entry. The parser must raise a value error whose message
    # starts a "steps: " detail.
    path = _make_fixture(
        f"""
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/x",
                         steps={steps_literal},
                         constraint={{"rel_up": 0.2, "rel_down": 0.2}})
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="steps: "):
        parse_ci_gate_specs(path)


def test_direction_key_is_gone(tmp_path):
    # [FUNCTION DOC] A constraint still carrying the removed one-sided "direction" key makes the
    # parser raise a value error "unknown key 'direction'" (constraints are always two-sided).
    # The one-sided direction abstraction was removed: every constraint is
    # two-sided, a lenient side gets a wide band. A stale declaration must
    # fail loud, not silently drop its direction.
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/x",
                         steps="last",
                         constraint={"rel_up": 0.2, "rel_down": 0.8, "direction": "higher_is_worse"})
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="unknown key 'direction'"):
        parse_ci_gate_specs(path)


def test_negative_rel_rejected(tmp_path):
    # [FUNCTION DOC] A negative relative upper band makes the parser raise a value error naming
    # param 'rel_up'.
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/x",
                         steps="last", constraint={"rel_up": -0.2, "rel_down": 0.2})
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="param 'rel_up'"):
        parse_ci_gate_specs(path)


def test_duplicate_dict_key_rejected(tmp_path):
    # [FUNCTION DOC] A constraint dictionary literal that repeats the key 'rel_up' makes the parser
    # raise a value error "duplicate key 'rel_up'" (a plain dict would silently keep the last).
    # A plain dict would silently keep the last value; the parser must reject.
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/x",
                         steps="last",
                         constraint={"rel_up": 0.2, "rel_up": 0.3, "rel_down": 0.2})
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="duplicate key 'rel_up'"):
        parse_ci_gate_specs(path)


def test_sub_label_argument_is_gone(tmp_path):
    # [FUNCTION DOC] A declaration still passing the removed sub-label argument makes the parser
    # raise a value error "unknown argument 'sub_label'".
    # The old author-label argument was removed with the encoded-coordinate
    # design; a declaration still passing it must fail loud, not silently drop.
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/x",
                         steps="last", constraint={"rel_up": 0.2, "rel_down": 0.2},
                         sub_label="shard-0")
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="unknown argument 'sub_label'"):
        parse_ci_gate_specs(path)


def test_non_bool_enforce_rejected(tmp_path):
    # [FUNCTION DOC] An enforce argument given as the integer 1 makes the parser raise a value error
    # "enforce must be a boolean".
    path = _make_fixture(
        """
        from tests.ci.metric_history import register_ci_gate
        register_ci_gate(metric_key="train/x",
                         steps="last", constraint={"rel_up": 0.2, "rel_down": 0.2}, enforce=1)
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="enforce must be a boolean"):
        parse_ci_gate_specs(path)


def test_register_ci_gate_does_not_disturb_suite_parsing(tmp_path):
    # [FUNCTION DOC] A fixture holding one CUDA suite registration plus one gate declaration is run
    # through the suite registry parser; asserts exactly one registry entry, of the registry type,
    # with the CUDA backend and the declared 8-GPU H100 suite (the gate call is ignored).
    # The suite RegistryVisitor must still find exactly the register_cuda_ci
    # call and ignore the register_ci_gate calls beside it.
    path = _make_fixture(
        """
        from tests.ci.ci_register import register_cuda_ci
        from tests.ci.metric_history import register_ci_gate
        register_cuda_ci(est_time=600, suite="stage-c-8-gpu-h100", labels=["megatron"], hardware=["hopper"])
        register_ci_gate(metric_key="train/grad_norm",
                         steps="all", constraint={"rel_up": 0.2, "rel_down": 0.2})
        """,
        tmp_path,
    )
    registries = ut_parse_one_file(path)
    assert len(registries) == 1
    assert isinstance(registries[0], CIRegistry)
    assert registries[0].backend == HWBackend.CUDA
    assert registries[0].suite == "stage-c-8-gpu-h100"


def test_register_ci_gate_runtime_is_noop():
    # [FUNCTION DOC] Calling the gate registration function at runtime with the full set of
    # arguments succeeds and returns nothing.
    from tests.ci.metric_history import register_ci_gate

    assert (
        register_ci_gate(
            metric_key="train/grad_norm",
            steps="all",
            constraint={"rel_up": 0.2, "rel_down": 0.2},
        )
        is None
    )
