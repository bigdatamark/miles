# [MODULE DOC] Tests for the parallel configuration helpers of the run_megatron tool
# (miles.utils.debug_utils.run_megatron.cli.parallel_utils): the immutable ParallelConfig (tensor,
# pipeline, context, expert and expert-tensor parallel sizes; process count is the product of tensor,
# pipeline and context sizes; effective expert size falls back to tensor size when unset; the process
# count must be divisible by the effective expert size), its directory name rendering, construction
# from parsed dictionaries or run arguments, string form, and the parser of a flag string.
from __future__ import annotations

import dataclasses

import pytest

from miles.utils.debug_utils.run_megatron.cli.parallel_utils import ParallelConfig, parse_parallel_args


class TestParallelConfigDefaults:
    # [CLASS DOC] Tests of default values and immutability.
    def test_frozen(self) -> None:
        # [FUNCTION DOC] Assigning to a field of a config raises the dataclass frozen-instance error.
        config = ParallelConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.tp = 2  # type: ignore[misc]

    def test_defaults(self) -> None:
        # [FUNCTION DOC] Defaults: tensor, pipeline, context sizes 1, expert size unset, expert
        # tensor size 1.
        config = ParallelConfig()
        assert config.tp == 1
        assert config.pp == 1
        assert config.cp == 1
        assert config.ep is None
        assert config.etp == 1


class TestEffectiveEp:
    # [CLASS DOC] Tests of the effective expert parallel size property.
    def test_ep_none_falls_back_to_tp(self) -> None:
        # [FUNCTION DOC] Unset expert size with tensor size 4 gives effective size 4.
        config = ParallelConfig(tp=4)
        assert config.effective_ep == 4

    def test_ep_explicit(self) -> None:
        # [FUNCTION DOC] Explicit expert size 2 with tensor size 4 gives effective size 2.
        config = ParallelConfig(tp=4, ep=2)
        assert config.effective_ep == 2

    def test_ep_1_is_not_none(self) -> None:
        # [FUNCTION DOC] Explicit expert size 1 is respected (not treated as unset): effective size 1.
        config = ParallelConfig(tp=4, ep=1)
        assert config.effective_ep == 1


class TestNproc:
    # [CLASS DOC] Tests of the process count property.
    def test_all_ones(self) -> None:
        # [FUNCTION DOC] The default config has process count 1.
        assert ParallelConfig().nproc == 1

    def test_tp_pp_cp(self) -> None:
        # [FUNCTION DOC] Tensor 2, pipeline 3, context 4 give process count 24 (the product).
        config = ParallelConfig(tp=2, pp=3, cp=4)
        assert config.nproc == 24

    def test_ep_etp_dont_affect_nproc(self) -> None:
        # [FUNCTION DOC] Expert sizes do not change the process count: tensor 2 gives 2.
        config = ParallelConfig(tp=2, ep=2, etp=2)
        assert config.nproc == 2


class TestPostInit:
    # [CLASS DOC] Tests of the construction-time validation (process count divisible by the
    # effective expert size).
    def test_valid_nproc_divisible_by_ep(self) -> None:
        # [FUNCTION DOC] Tensor 4 with expert 2 (process count 4) constructs without error.
        ParallelConfig(tp=4, pp=1, cp=1, ep=2)

    def test_invalid_nproc_not_divisible_by_ep(self) -> None:
        # [FUNCTION DOC] Tensor 3 with expert 2 (process count 3) raises a value error saying the
        # count is not divisible by the effective expert size.
        with pytest.raises(ValueError, match="not divisible by effective EP"):
            ParallelConfig(tp=3, pp=1, cp=1, ep=2)

    def test_ep_none_uses_tp_for_validation(self) -> None:
        # [FUNCTION DOC] Tensor 2 with unset expert size is valid (count 2 divisible by effective 2).
        # nproc=tp*pp*cp=2, effective_ep=tp=2, 2%2=0 → valid
        ParallelConfig(tp=2)

    def test_ep_none_invalid(self) -> None:
        # [FUNCTION DOC] Constructs two configs with unset expert size (tensor 2 with pipeline 3, and
        # tensor 3); both must construct without error.
        # KNOWN ISSUE: the test name says "invalid" but both cases are valid and nothing is asserted
        # about an invalid case.
        # nproc=tp*pp*cp=2*3*1=6, effective_ep=tp=2, 6%2=0 → valid
        ParallelConfig(tp=2, pp=3)
        # nproc=3*1*1=3, effective_ep=3, 3%3=0 → valid
        ParallelConfig(tp=3)


class TestDirName:
    # [CLASS DOC] Tests of the directory name rendering: fields are joined with underscores in the
    # order tp, pp, cp, ep, etp; a field is omitted when it has its skip value (pp, cp, etp equal 1;
    # ep unset or equal to tp); tp is always present.
    def test_tp_only(self) -> None:
        # [FUNCTION DOC] Tensor 2 alone renders "tp2".
        # tp is never skipped (skip value is None), pp=1 skipped, cp=1 skipped,
        # ep=None skipped, etp=1 skipped
        name = ParallelConfig(tp=2).dir_name()
        assert name == "tp2"

    def test_tp_cp(self) -> None:
        # [FUNCTION DOC] Tensor 2 and context 4 render "tp2_cp4".
        name = ParallelConfig(tp=2, cp=4).dir_name()
        assert name == "tp2_cp4"

    def test_ep_equals_tp_is_skipped(self) -> None:
        # [FUNCTION DOC] Expert size equal to tensor size is omitted from the name.
        # ep == tp → skipped
        name = ParallelConfig(tp=4, ep=4).dir_name()
        assert "ep" not in name

    def test_ep_differs_from_tp(self) -> None:
        # [FUNCTION DOC] Expert size 2 with tensor size 4 appears as "ep2" in the name.
        name = ParallelConfig(tp=4, ep=2).dir_name()
        assert "ep2" in name

    def test_pp_1_skipped(self) -> None:
        # [FUNCTION DOC] Pipeline size 1 is omitted from the name.
        name = ParallelConfig(tp=2, pp=1).dir_name()
        assert "pp" not in name

    def test_pp_gt1_included(self) -> None:
        # [FUNCTION DOC] Pipeline size 2 appears as "pp2".
        name = ParallelConfig(tp=2, pp=2).dir_name()
        assert "pp2" in name

    def test_etp_gt1_included(self) -> None:
        # [FUNCTION DOC] Expert tensor size 2 appears as "etp2".
        name = ParallelConfig(tp=2, etp=2).dir_name()
        assert "etp2" in name

    def test_all_fields(self) -> None:
        # [FUNCTION DOC] All fields non-default render exactly "tp2_pp3_cp4_ep8_etp2".
        name = ParallelConfig(tp=2, pp=3, cp=4, ep=8, etp=2).dir_name()
        assert name == "tp2_pp3_cp4_ep8_etp2"

    def test_default_config(self) -> None:
        # [FUNCTION DOC] The default config renders exactly "tp1".
        # tp=1, pp=1, cp=1, ep=None, etp=1
        # tp=1 is not None so included; pp=1 skipped; cp=1 skipped; ep=None skipped; etp=1 skipped
        name = ParallelConfig().dir_name()
        assert name == "tp1"


class TestFromParsedArgs:
    # [CLASS DOC] Tests of building a config from a dictionary of parsed values.
    def test_full_args(self) -> None:
        # [FUNCTION DOC] A dictionary with all five sizes sets all five fields.
        parsed = {"tp": 2, "pp": 3, "cp": 4, "ep": 8, "etp": 2}
        config = ParallelConfig.from_parsed_args(parsed)
        assert config.tp == 2
        assert config.pp == 3
        assert config.cp == 4
        assert config.ep == 8
        assert config.etp == 2

    def test_partial_args_use_defaults(self) -> None:
        # [FUNCTION DOC] A dictionary with only tensor size 4 leaves the other fields at defaults.
        parsed = {"tp": 4}
        config = ParallelConfig.from_parsed_args(parsed)
        assert config.tp == 4
        assert config.pp == 1
        assert config.cp == 1
        assert config.ep is None
        assert config.etp == 1

    def test_empty_args(self) -> None:
        # [FUNCTION DOC] An empty dictionary yields a config equal to the default config.
        config = ParallelConfig.from_parsed_args({})
        assert config == ParallelConfig()


class TestFromRunArgs:
    # [CLASS DOC] Tests of building a config from run command arguments.
    def test_extracts_fields(self) -> None:
        # [FUNCTION DOC] Builds run arguments (model type, checkpoint, tp 4, pp 2, cp 2, ep 8, etp 2;
        # the run args class is imported lazily) and asserts the config copies all five sizes.
        from miles.utils.debug_utils.run_megatron.cli.commands.args import RunArgs

        args = RunArgs(
            model_type="deepseek_v3",
            hf_checkpoint="/fake",  # type: ignore[arg-type]
            tp=4,
            pp=2,
            cp=2,
            ep=8,
            etp=2,
        )
        config = ParallelConfig.from_run_args(args)
        assert config.tp == 4
        assert config.pp == 2
        assert config.cp == 2
        assert config.ep == 8
        assert config.etp == 2


class TestStr:
    # [CLASS DOC] Tests of the human-readable string form.
    def test_contains_all_fields(self) -> None:
        # [FUNCTION DOC] The string of a fully specified config contains "tp=2", "pp=3", "cp=4",
        # "ep=8", "etp=2" and "nproc=24".
        config = ParallelConfig(tp=2, pp=3, cp=4, ep=8, etp=2)
        s = str(config)
        assert "tp=2" in s
        assert "pp=3" in s
        assert "cp=4" in s
        assert "ep=8" in s
        assert "etp=2" in s
        assert "nproc=24" in s


class TestParseParallelArgs:
    # [CLASS DOC] Tests of parsing a flag string such as "--tp 2 --cp 4" into a dictionary of integers.
    def test_basic(self) -> None:
        # [FUNCTION DOC] "--tp 2 --cp 4" parses to a mapping of exactly tp 2 and cp 4.
        result = parse_parallel_args("--tp 2 --cp 4")
        assert result == {"tp": 2, "cp": 4}

    def test_all_flags(self) -> None:
        # [FUNCTION DOC] All five flags parse to a mapping with all five integer values.
        result = parse_parallel_args("--tp 2 --pp 3 --cp 4 --ep 8 --etp 2")
        assert result == {"tp": 2, "pp": 3, "cp": 4, "ep": 8, "etp": 2}

    def test_empty_string(self) -> None:
        # [FUNCTION DOC] An empty string parses to an empty mapping.
        result = parse_parallel_args("")
        assert result == {}

    def test_omitted_flags_excluded(self) -> None:
        # [FUNCTION DOC] Flags not given (pp, cp) are absent from the result of parsing "--tp 2".
        result = parse_parallel_args("--tp 2")
        assert "pp" not in result
        assert "cp" not in result
