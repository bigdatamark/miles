# [MODULE DOC]
# Tests for the parallelism configuration used by the Megatron debug runner CLI. The config is a
# frozen dataclass with tensor, pipeline, context, expert (optional, defaults to the tensor size)
# and expert-tensor parallel sizes; process count is tp*pp*cp, and construction raises a value
# error when that count is not divisible by the effective expert size. Also tested: building from
# a parsed-args mapping or run args (missing keys take defaults), the string form, the directory
# name form (omits default pp/cp/etp and an expert size equal to tp), and parsing a command-line
# style string of parallel flags into a mapping of ints.
import dataclasses

import pytest

from miles.utils.debug_utils.run_megatron.cli.parallel_utils import ParallelConfig, parse_parallel_args

# ---------------------------------------------------------------------------
# Construction & defaults
# ---------------------------------------------------------------------------


class TestConstruction:
    # [CLASS DOC] Tests of defaults and immutability.
    def test_defaults(self) -> None:
        # [FUNCTION DOC] Default config: tp, pp, cp, etp are 1 and ep is none.
        config = ParallelConfig()
        assert config.tp == 1
        assert config.pp == 1
        assert config.cp == 1
        assert config.ep is None
        assert config.etp == 1

    def test_explicit_values(self) -> None:
        # [FUNCTION DOC] Explicit sizes are stored unchanged.
        config = ParallelConfig(tp=2, pp=4, cp=2, ep=8, etp=2)
        assert config.tp == 2
        assert config.pp == 4
        assert config.cp == 2
        assert config.ep == 8
        assert config.etp == 2

    def test_frozen(self) -> None:
        # [FUNCTION DOC] Assigning a field raises the dataclass frozen-instance error.
        config = ParallelConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.tp = 2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    # [CLASS DOC] Tests of derived properties.
    def test_nproc(self) -> None:
        # [FUNCTION DOC] Process count for tp 2, pp 4, cp 2 is 16.
        assert ParallelConfig(tp=2, pp=4, cp=2).nproc == 16

    def test_nproc_defaults(self) -> None:
        # [FUNCTION DOC] Default process count is 1.
        assert ParallelConfig().nproc == 1

    def test_effective_ep_defaults_to_tp(self) -> None:
        # [FUNCTION DOC] With ep none and tp 4, the effective expert size is 4.
        config = ParallelConfig(tp=4)
        assert config.effective_ep == 4

    def test_effective_ep_explicit(self) -> None:
        # [FUNCTION DOC] With ep 2 and tp 4, the effective expert size is 2.
        config = ParallelConfig(tp=4, ep=2)
        assert config.effective_ep == 2


# ---------------------------------------------------------------------------
# __post_init__ validation
# ---------------------------------------------------------------------------


class TestValidation:
    # [CLASS DOC] Tests of construction-time validation.
    def test_valid_config_no_error(self) -> None:
        # [FUNCTION DOC] tp 2, pp 2, cp 2, ep 4 (8 processes divisible by 4) constructs without error.
        ParallelConfig(tp=2, pp=2, cp=2, ep=4)

    def test_nproc_not_divisible_by_ep_raises(self) -> None:
        # [FUNCTION DOC] tp 2 with ep 3 (2 processes) raises a value error saying the process count
        # is not divisible by the effective expert size.
        with pytest.raises(ValueError, match="not divisible by effective EP"):
            ParallelConfig(tp=2, pp=1, cp=1, ep=3)

    def test_default_ep_equals_tp_always_valid(self) -> None:
        # [FUNCTION DOC] With ep unset, any tp/pp/cp (here 3, 2, 1) constructs without error.
        # When ep=None, effective_ep=tp, and nproc=tp*pp*cp is always
        # divisible by tp, so validation never fires.
        ParallelConfig(tp=3, pp=2, cp=1)  # no error


# ---------------------------------------------------------------------------
# from_parsed_args
# ---------------------------------------------------------------------------


class TestFromParsedArgs:
    # [CLASS DOC] Tests of building a config from a mapping of parsed ints.
    def test_all_provided(self) -> None:
        # [FUNCTION DOC] A mapping with all five keys gives an equal explicit config.
        config = ParallelConfig.from_parsed_args({"tp": 2, "pp": 4, "cp": 2, "ep": 8, "etp": 2})
        assert config == ParallelConfig(tp=2, pp=4, cp=2, ep=8, etp=2)

    def test_partial_uses_defaults(self) -> None:
        # [FUNCTION DOC] A mapping with only tp 4 gives tp 4 and defaults for the rest.
        config = ParallelConfig.from_parsed_args({"tp": 4})
        assert config == ParallelConfig(tp=4, pp=1, cp=1, ep=None, etp=1)

    def test_empty_dict_gives_defaults(self) -> None:
        # [FUNCTION DOC] An empty mapping gives the default config.
        config = ParallelConfig.from_parsed_args({})
        assert config == ParallelConfig()


# ---------------------------------------------------------------------------
# from_run_args
# ---------------------------------------------------------------------------


class TestFromRunArgs:
    # [CLASS DOC] Tests of building a config from the runner's argument object.
    def test_from_run_args(self) -> None:
        # [FUNCTION DOC] Run args with model type, checkpoint path and tp 2, pp 4, cp 1, ep 2, etp 1
        # give the matching config.
        from miles.utils.debug_utils.run_megatron.cli.commands.args import RunArgs

        args = RunArgs(
            model_type="test",
            hf_checkpoint="/tmp/hf",
            tp=2,
            pp=4,
            cp=1,
            ep=2,
            etp=1,
        )
        config = ParallelConfig.from_run_args(args)
        assert config == ParallelConfig(tp=2, pp=4, cp=1, ep=2, etp=1)

    def test_from_run_args_defaults(self) -> None:
        # [FUNCTION DOC] Run args with no parallel sizes give the default config.
        from miles.utils.debug_utils.run_megatron.cli.commands.args import RunArgs

        args = RunArgs(
            model_type="test",
            hf_checkpoint="/tmp/hf",
        )
        config = ParallelConfig.from_run_args(args)
        assert config == ParallelConfig()


# ---------------------------------------------------------------------------
# __str__
# ---------------------------------------------------------------------------


class TestStr:
    # [CLASS DOC] Tests of the string form.
    def test_str_repr(self) -> None:
        # [FUNCTION DOC] The string of a full config contains each size as name=value and nproc=16.
        config = ParallelConfig(tp=2, pp=4, cp=2, ep=8, etp=2)
        text = str(config)
        assert "tp=2" in text
        assert "pp=4" in text
        assert "cp=2" in text
        assert "ep=8" in text
        assert "etp=2" in text
        assert "nproc=16" in text

    def test_str_defaults(self) -> None:
        # [FUNCTION DOC] The default config string contains nproc=1.
        text = str(ParallelConfig())
        assert "nproc=1" in text


# ---------------------------------------------------------------------------
# dir_name
# ---------------------------------------------------------------------------


class TestDirName:
    # [CLASS DOC] Tests of the directory-name form: tp always, then pp, cp, ep, etp only when they
    # differ from their default (ep omitted when equal to tp), joined by underscores.
    def test_tp_only(self) -> None:
        # [FUNCTION DOC] tp 2 gives "tp2".
        assert ParallelConfig(tp=2).dir_name() == "tp2"

    def test_all_non_default(self) -> None:
        # [FUNCTION DOC] All sizes non-default gives "tp2_pp2_cp2_ep4_etp2".
        assert ParallelConfig(tp=2, pp=2, cp=2, ep=4, etp=2).dir_name() == "tp2_pp2_cp2_ep4_etp2"

    def test_ep_equal_tp_omitted(self) -> None:
        # [FUNCTION DOC] ep equal to tp (4) is omitted: "tp4".
        assert ParallelConfig(tp=4, ep=4).dir_name() == "tp4"

    def test_ep_different_from_tp_included(self) -> None:
        # [FUNCTION DOC] tp 4 with ep 2 gives "tp4_ep2".
        assert ParallelConfig(tp=4, ep=2).dir_name() == "tp4_ep2"

    def test_pp1_cp1_omitted(self) -> None:
        # [FUNCTION DOC] pp 1 and cp 1 are omitted: "tp2".
        assert ParallelConfig(tp=2, pp=1, cp=1).dir_name() == "tp2"

    def test_defaults_dir_name(self) -> None:
        # [FUNCTION DOC] The default config gives "tp1".
        assert ParallelConfig().dir_name() == "tp1"


# ---------------------------------------------------------------------------
# parse_parallel_args
# ---------------------------------------------------------------------------


class TestParseParallelArgs:
    # [CLASS DOC] Tests of parsing a flags string into a mapping of present keys to ints.
    def test_full_string(self) -> None:
        # [FUNCTION DOC] All five flags parse into a mapping of five ints.
        result = parse_parallel_args("--tp 2 --pp 4 --cp 2 --ep 8 --etp 2")
        assert result == {"tp": 2, "pp": 4, "cp": 2, "ep": 8, "etp": 2}

    def test_partial_string(self) -> None:
        # [FUNCTION DOC] Only the flags present appear in the mapping (tp and cp).
        result = parse_parallel_args("--tp 4 --cp 2")
        assert result == {"tp": 4, "cp": 2}

    def test_empty_string(self) -> None:
        # [FUNCTION DOC] An empty string gives an empty mapping.
        result = parse_parallel_args("")
        assert result == {}

    def test_roundtrip_with_from_parsed_args(self) -> None:
        # [FUNCTION DOC] Parsing "--tp 2 --pp 4 --cp 2" and building a config from the result equals
        # the directly built config.
        parsed = parse_parallel_args("--tp 2 --pp 4 --cp 2")
        config = ParallelConfig.from_parsed_args(parsed)
        assert config == ParallelConfig(tp=2, pp=4, cp=2)
