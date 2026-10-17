# [MODULE DOC] HELPER: Megatron model-argument definition for a 4-layer partial of MiMo-V2.6-Flash
# (layers 0, 1, 5, 6 of the full model, covering every decoder variant), derived from the full
# definition. Import has no side effects.
from model_args_utils import load_sibling_model_args


def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the sibling mimo-v2.6-flash flag string with 4 layers.
    # 4-layer partial from `tools/convert_mimo_v2_to_bf16.py --layers 0,1,5,6`:
    # global+dense, SWA+MoE, global+MoE, SWA+MoE (every decoder variant, PP=2 splits it in half).
    return load_sibling_model_args(__file__, "mimo-v2.6-flash", nlayers=4)
