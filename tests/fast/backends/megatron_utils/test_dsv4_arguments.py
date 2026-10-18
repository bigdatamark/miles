# [MODULE DOC] CPU tests of the DeepSeek-V4 command-line argument helpers: detection of a V4 model
# by its layer-spec module, and normalization of the implementation choice (default and "megatron"
# select the hybrid attention variant with hyper-connections enabled; "miles" selects the plain
# variant), with parse-time value errors for unsupported combinations (megatron implementation with
# tensor parallel size above 1 or the tilelang kernel backend; miles implementation with the cuDNN
# kernel backend).
from argparse import ArgumentParser, Namespace

import pytest

from miles_plugins.models.deepseek_v4.arguments import (
    DSV4_SPEC_MODULE,
    add_dsv4_arguments,
    is_dsv4_model,
    normalize_dsv4_args,
)


def _parse(*argv: str) -> Namespace:
    # [FUNCTION DOC] HELPER: Builds a parser with the V4 arguments registered plus a tensor-parallel
    # size (default 1), a kernel backend (default none) and a spec option (default: the V4 spec module
    # and its getter name), and returns the namespace parsed from the given command-line tokens.
    parser = ArgumentParser()
    add_dsv4_arguments(parser)
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1)
    parser.add_argument("--dsa-kernel-backend", default=None)
    parser.add_argument("--spec", nargs="*", default=[DSV4_SPEC_MODULE, "get_dsv4_spec"])
    return parser.parse_args(argv)


def test_only_the_dsv4_spec_triggers_normalization():
    # [FUNCTION DOC] Asserts the V4 detector is true for the default spec, false for another model's
    # spec list, and false when the spec is none.
    assert is_dsv4_model(_parse())

    other = _parse()
    other.spec = ["miles_plugins.models.glm5.glm5", "get_glm5_spec"]
    assert not is_dsv4_model(other)

    unspecced = _parse()
    unspecced.spec = None
    assert not is_dsv4_model(unspecced)


def test_impl_selects_the_attention_variant():
    # [FUNCTION DOC] After normalization, the default implementation gives experimental attention
    # variant "dsv4_hybrid"; "miles" gives "dsv4"; "megatron" gives "dsv4_hybrid" and turns on
    # hyper-connections.
    default = _parse()
    normalize_dsv4_args(default)
    assert default.experimental_attention_variant == "dsv4_hybrid"

    miles = _parse("--dsv4-impl", "miles")
    normalize_dsv4_args(miles)
    assert miles.experimental_attention_variant == "dsv4"

    megatron = _parse("--dsv4-impl", "megatron")
    normalize_dsv4_args(megatron)
    assert megatron.experimental_attention_variant == "dsv4_hybrid"
    assert megatron.enable_hyper_connections


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (("--dsv4-impl", "megatron", "--tensor-model-parallel-size", "8"), "tensor-model-parallel-size 1"),
        (("--dsv4-impl", "megatron", "--dsa-kernel-backend", "tilelang"), "does not support"),
        (("--dsv4-impl", "miles", "--dsa-kernel-backend", "cudnn"), "ignores cuDNN"),
    ],
    ids=["megatron-needs-tp1", "megatron-rejects-tilelang", "miles-rejects-cudnn"],
)
def test_unsupported_combinations_fail_at_parse_time(argv, message):
    # [FUNCTION DOC] Parametrized over three invalid combinations; asserts normalization raises a
    # value error whose message matches the expected text.
    """Megatron would only assert deep inside config post-init, or silently mis-run."""
    with pytest.raises(ValueError, match=message):
        normalize_dsv4_args(_parse(*argv))
