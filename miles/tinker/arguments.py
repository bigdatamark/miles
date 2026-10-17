# [MODULE DOC]
# Launch arguments for the Tinker-compatible gateway: registration of its command-line flags and
# a post-parse configuration step that checks compatibility with the training setup and derives the
# LoRA target groups the gateway trains.
from miles.utils.lora.hf_lora_targets import LORA_TARGET_GROUPS, parse_lora_targets


def add_tinker_arguments(parser):
    # [FUNCTION DOC] Add an argument group titled Tinker to the parser (mutates it) with flags
    # prefixed by tinker-: server host (default all interfaces), server port (integer, default
    # 10613), base model name advertised by the gateway (no default here; documented as falling
    # back to the HF checkpoint) and checkpoint root directory (documented as defaulting to a
    # tinker folder under the save directory; the defaults are applied elsewhere). Returns the
    # parser.
    group = parser.add_argument_group("Tinker")

    def add_argument(name, **kwargs):
        # [FUNCTION DOC] HELPER: Add an option named with the tinker prefix and the given name to
        # the group, forwarding all keyword settings; returns the created action.
        return group.add_argument(f"--tinker-{name}", **kwargs)

    add_argument("server-host", default="0.0.0.0")
    add_argument("server-port", type=int, default=10613)
    add_argument(
        "base-model",
        help="Model name advertised by the gateway (default: --hf-checkpoint)",
    )
    add_argument(
        "checkpoint-root",
        help="Directory for tinker:// checkpoints (default: <save>/tinker)",
    )
    return parser


def configure_tinker_args(args):
    # [FUNCTION DOC]
    # Validate and derive Tinker settings on the parsed args (mutates them). Assertion errors if
    # the training backend is not megatron or the exclude-modules option is set. The target modules
    # option is parsed into groups; when unset all known LoRA target groups are used; any group
    # outside the known set (attention, MLP, unembed) fails an assertion. Stores the group list
    # as the Tinker LoRA groups attribute and overwrites the target modules option with it.
    assert args.train_backend == "megatron", "Tinker requires the Megatron backend"
    assert args.exclude_modules is None, "Tinker selects complete training groups; --exclude-modules is not supported"
    groups = parse_lora_targets(args.target_modules)
    if groups is None:
        groups = list(LORA_TARGET_GROUPS)
    assert set(groups) <= set(LORA_TARGET_GROUPS), "Tinker --target-modules accepts only attn,mlp,unembed groups"
    args.tinker_lora_groups = groups
    args.target_modules = groups
