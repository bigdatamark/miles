# [PACKAGE DOC]
# Megatron-LM training backend: the Ray train actor, model/optimizer construction, checkpointing,
# weight conversion between Megatron and Hugging Face layouts, LoRA support, and fault-tolerance
# helpers. Importing the package has global side effects, performed at import time in this order:
# (1) if the deep_ep package and the GPU memory saver are importable, the deep_ep communication
# buffer constructor is wrapped so that memory-saver tracking is paused while the buffer is
# allocated, the GPU is synchronized afterwards, and tracking is resumed (otherwise a warning is
# logged); (2) the optional megatron bridge plugin package is imported, with any failure logged as
# a warning; (3) the megatron logger level is set to warning.
import logging

import torch

try:
    import deep_ep
    from torch_memory_saver import torch_memory_saver

    old_init = deep_ep.Buffer.__init__

    def new_init(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Replacement constructor for the deep_ep buffer. If the memory
        # saver has an active implementation, turns off its "interesting region" tracking through
        # the native library before running the original constructor, then synchronizes the CUDA
        # device and turns tracking back on, so the buffer's allocations are not captured by
        # offload/pause. Arguments are passed through unchanged. Nothing is returned.
        if torch_memory_saver._impl is not None:
            torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(False)
        old_init(self, *args, **kwargs)
        torch.cuda.synchronize()
        if torch_memory_saver._impl is not None:
            torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(True)

    deep_ep.Buffer.__init__ = new_init
except ImportError:
    logging.warning("deep_ep is not installed, some functionalities may be limited.")

try:
    import miles_plugins.megatron_bridge  # noqa: F401
except Exception as _e:  # best-effort; not every environment uses megatron.bridge
    logging.warning("miles megatron.bridge plugins failed to load: %s", _e)

logging.getLogger("megatron").setLevel(logging.WARNING)
