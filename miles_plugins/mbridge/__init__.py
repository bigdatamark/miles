# [PACKAGE DOC]
# Weight-conversion bridges for the mbridge library, one module per model family. Importing this
# package imports every bridge module, which registers each bridge class under its model type name
# (through the library's register decorator) and re-exports the main bridge classes. The listed
# public names are the main bridges of each module; helper bridges defined in the modules (for
# example the glm moe dsa bridge in the deepseek v32 module) are registered by import without being
# listed.
from .deepseek_v4 import DeepseekV4Bridge
from .deepseek_v4_1 import DeepseekV41Bridge
from .deepseek_v32 import DeepseekV32Bridge
from .glm4 import GLM4Bridge
from .glm4moe import GLM4MoEBridge
from .glm4moe_lite import GLM4MoELiteBridge
from .glm5_next import Glm5NextBridge
from .inkling import InklingBridge
from .joyai_llm_flash import JoyAILLMFlashBridge
from .kimi_k3 import KimiK3Bridge
from .mimo import MimoBridge
from .qwen3_5 import Qwen3_5Bridge
from .qwen3_8_next import Qwen38NextBridge
from .qwen3_next import Qwen3NextBridge

__all__ = [
    "GLM4Bridge",
    "GLM4MoEBridge",
    "GLM4MoELiteBridge",
    "Qwen3NextBridge",
    "Qwen3_5Bridge",
    "Qwen38NextBridge",
    "MimoBridge",
    "DeepseekV32Bridge",
    "Glm5NextBridge",
    "DeepseekV4Bridge",
    "DeepseekV41Bridge",
    "JoyAILLMFlashBridge",
    "InklingBridge",
    "KimiK3Bridge",
]
