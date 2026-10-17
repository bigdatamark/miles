# [PACKAGE DOC]
# Hub-kernel plugin: resolves module-level compute kernels (linear-attention, causal convolution,
# varlen flash attention) from Hugging Face Hub repos through the optional kernels package and
# binds them onto model instances. loader holds the spec type and the collective resolution state,
# presets the default slot-to-repo mapping, and binders the per-architecture rebinding. The package
# re-exports the kernel-state class and the kernel-spec type.
from miles.backends.fsdp_utils.plugins.hf_kernels.loader import HubKernels, HubKernelSpec

__all__ = ["HubKernels", "HubKernelSpec"]
