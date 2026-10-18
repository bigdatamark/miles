# [MODULE DOC] HELPER: Small utility module deciding how checkpoint tensor names should be
# renamed when converting checkpoints: DeepSeek-V4 native-format checkpoints use the upstream
# renaming function of the serving library's model class, and everything else uses an identity
# mapping. Used by the FP8-to-BF16 conversion tool.
import json
from collections.abc import Callable


def _is_deepseek_v4_native(config_path: str, weight_map: dict) -> bool:
    # [FUNCTION DOC] HELPER: Reads the config file; true only if its architectures list contains
    # the DeepSeek-V4 causal-LM name and the weight map has a tensor named "embed.weight" (the
    # native-format marker). Missing architectures key counts as empty; I/O errors propagate.
    # Only native-format V4 checkpoints need the HF remap; re-saved HF-format V4
    # and V3/V3.2 checkpoints are already HF-named (embed.weight is the native marker).
    with open(config_path) as f:
        is_v4_arch = "DeepseekV4ForCausalLM" in json.load(f).get("architectures", [])
    return is_v4_arch and "embed.weight" in weight_map


def get_param_name_remap(config_path: str, weight_map: dict) -> Callable[[str], str]:
    # [FUNCTION DOC] Returns a function mapping a tensor name to the name to use. For native
    # DeepSeek-V4 checkpoints, lazily imports the serving library's V4 model class and returns
    # its name-remapping function (raises if the library is not installed); otherwise returns a
    # function that returns the name unchanged.
    # DeepSeek-V4 native checkpoints remap to HF names; everything else is identity.
    if _is_deepseek_v4_native(config_path, weight_map):
        from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

        return DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format
    return lambda name: name
