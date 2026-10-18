# [MODULE DOC]
# mbridge bridge alias for the JoyAI-LLM-Flash model type, which has the DeepSeek V3 architecture.
# It only registers the new model type name and turns multi-token prediction off in the config.
from mbridge.core import register_model
from mbridge.models import DeepseekV3Bridge


@register_model("joyai_llm_flash")
class JoyAILLMFlashBridge(DeepseekV3Bridge):
    # [CLASS DOC]
    # Registered under the name joyai_llm_flash; behaves exactly as the V3 bridge except for the
    # config override below.
    """Minimal JoyAI alias bridge with MTP explicitly disabled.

    JoyAI-LLM-Flash exposes `model_type=joyai_llm_flash` while architecture is
    DeepSeek-V3 compatible. We only register the model type alias so
    AutoBridge can instantiate. MTP conversion is intentionally disabled for now.
    """

    def _build_config(self):
        # [FUNCTION DOC] Build the config with the parent builder, then set the multi-token prediction
        # layer count to none: as an attribute if the config object has one, or as a key if the
        # result is a plain mapping that lacks the attribute. Returns the config.
        config = super()._build_config()

        # Keep JoyAI conversion on the non-MTP path for now.
        if hasattr(config, "mtp_num_layers"):
            config.mtp_num_layers = None
        elif isinstance(config, dict):
            config["mtp_num_layers"] = None

        return config
