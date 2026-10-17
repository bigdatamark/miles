# [MODULE DOC] Single place that imports every SGLang dependency the Megatron actor needs, so the
# rest of the package imports them from here. Some FP8 helpers are optional: when an import fails
# the name is bound to none (UE8M0 quantization helpers and the deepgemm requant check together;
# per-block FP8 cast separately). The torch reductions monkey-patch helper and the tensor bucket
# class are imported from either their new or old SGLang locations (fallbacks), while the
# multiprocessing serializer has no fallback and a missing import raises. Exports are listed in
# the public names list.
# the file to manage all sglang deps in the megatron actor
try:
    from sglang.srt.layers.quantization.fp8_utils import quant_weight_ue8m0, transform_scale_ue8m0
    from sglang.srt.model_loader.utils import should_deepgemm_weight_requant_ue8m0
except ImportError:
    quant_weight_ue8m0 = None
    transform_scale_ue8m0 = None
    should_deepgemm_weight_requant_ue8m0 = None

try:
    from sglang.srt.layers.quantization.fp8_utils import per_block_cast_to_fp8
except ImportError:
    per_block_cast_to_fp8 = None

try:
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
except ImportError:
    from sglang.srt.patch_torch import monkey_patch_torch_reductions

from sglang.srt.utils import MultiprocessingSerializer

try:
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket  # type: ignore[import]
except ImportError:
    from sglang.srt.model_executor.model_runner import FlattenedTensorBucket  # type: ignore[import]

__all__ = [
    "per_block_cast_to_fp8",
    "quant_weight_ue8m0",
    "transform_scale_ue8m0",
    "should_deepgemm_weight_requant_ue8m0",
    "monkey_patch_torch_reductions",
    "MultiprocessingSerializer",
    "FlattenedTensorBucket",
]
