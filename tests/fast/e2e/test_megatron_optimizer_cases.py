# [MODULE DOC]
# Tests that the Megatron end-to-end case modules (qwen3 30B A3B, qwen3.5 35B A3B and glm4.7 flash
# families, one case each) render the right optimizer-related training arguments. Each case module
# and its family's shared common module are imported by name; the default wandb argument builder is
# stubbed to return nothing. Rendered argument strings are split shell-style into tokens. Muon
# cases must use the distributed muon optimizer with chunked state offload and no adam-only
# flags; the same case switched to adam keeps the cpu-offload flags; a plain muon optimizer name is
# rejected at case construction.
import importlib
import shlex
from dataclasses import replace

import pytest

from miles.utils.external_utils import command_utils

_CASES = (
    ("test_qwen3_30B_A3B", "test_nospec_nor3_bf16_sgl_tp4_meg_tp2cp2"),
    ("test_qwen3_5_35B_A3B", "test_nospec_nor3_bf16_sgl_dpattn2x2_meg_tp2pp2"),
    ("test_glm47_flash", "test_nospec_r3_bf16_deepep_sgl_dpattn2x2_meg_tp2cp2"),
)
_ADAM_ONLY_FLAGS = {
    "--optimizer-cpu-offload",
    "--overlap-cpu-optimizer-d2h-h2d",
    "--use-precision-aware-optimizer",
    "--rematerialize-param-from-master-weight",
    "--exp-avg-dtype",
    "--exp-avg-sq-dtype",
    "--main-params-dtype",
}


@pytest.fixture(params=_CASES, ids=[family for family, _ in _CASES])
def case_modules(request, monkeypatch):
    # [FUNCTION DOC] HELPER: Parametrized fixture over the three case families. Imports the
    # family's common module and its case module, takes the case module's case object, and stubs
    # the default wandb args function to return an empty string. Returns the pair (common module,
    # case).
    family, name = request.param
    common = importlib.import_module(f"tests.e2e.megatron.{family}._common")
    case = importlib.import_module(f"tests.e2e.megatron.{family}.{name}").CASE
    monkeypatch.setattr(command_utils, "get_default_wandb_args", lambda _: "")
    return common, case


@pytest.mark.parametrize("tight_host_memory", [False, True])
def test_muon_case_renders_compatible_distributed_offload(case_modules, monkeypatch, tight_host_memory):
    # [FUNCTION DOC] Optionally sets the common module's tight-host-memory switch (if it has one)
    # to the parametrized value, builds the train args for the case and asserts: exactly one
    # optimizer option whose value is dist_muon; the chunked optimizer state offload flag; offload
    # fraction 1.0; offload chunk size 1024 megabytes; none of the adam-only flags; and a rollout
    # count of 2.
    common, case = case_modules
    if hasattr(common, "TIGHT_HOST_MEMORY"):
        monkeypatch.setattr(common, "TIGHT_HOST_MEMORY", tight_host_memory)

    tokens = shlex.split(common.build_train_args(case, wandb_file=__file__))

    assert tokens.count("--optimizer") == 1
    assert tokens[tokens.index("--optimizer") + 1] == "dist_muon"
    assert "--chunked-optimizer-state-offload" in tokens
    assert tokens[tokens.index("--optimizer-state-offload-fraction") + 1] == "1.0"
    assert tokens[tokens.index("--optimizer-state-offload-chunk-size-mb") + 1] == "1024"
    assert not _ADAM_ONLY_FLAGS.intersection(tokens)
    assert tokens[tokens.index("--num-rollout") + 1] == "2"


@pytest.mark.parametrize("tight_host_memory", [False, True])
def test_adam_retains_its_existing_offload_path(case_modules, monkeypatch, tight_host_memory):
    # [FUNCTION DOC] Copies the case with the optimizer replaced by adam, builds train args (with
    # the tight-host-memory switch set when the module has one) and asserts: optimizer value adam;
    # cpu offload, overlapped offload transfer and precision-aware optimizer flags present; chunked
    # offload flag absent; for the qwen3 module family, the rematerialize-from-master-weight flag
    # present; and where the switch exists, the main params dtype option present exactly when tight
    # host memory is on.
    common, muon_case = case_modules
    if hasattr(common, "TIGHT_HOST_MEMORY"):
        monkeypatch.setattr(common, "TIGHT_HOST_MEMORY", tight_host_memory)

    case = replace(muon_case, optimizer="adam")
    tokens = shlex.split(common.build_train_args(case, wandb_file=__file__))

    assert tokens[tokens.index("--optimizer") + 1] == "adam"
    assert "--optimizer-cpu-offload" in tokens
    assert "--overlap-cpu-optimizer-d2h-h2d" in tokens
    assert "--use-precision-aware-optimizer" in tokens
    assert "--chunked-optimizer-state-offload" not in tokens
    if "qwen3" in common.__name__:
        assert "--rematerialize-param-from-master-weight" in tokens
    if hasattr(common, "TIGHT_HOST_MEMORY"):
        assert ("--main-params-dtype" in tokens) == tight_host_memory


def test_plain_muon_does_not_silently_select_the_adam_path(case_modules):
    # [FUNCTION DOC] Asserts that copying the case with the optimizer name muon raises a value
    # error saying the optimizer is unsupported.
    _, case = case_modules
    with pytest.raises(ValueError, match="unsupported optimizer"):
        replace(case, optimizer="muon")
