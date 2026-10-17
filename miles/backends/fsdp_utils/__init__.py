# [PACKAGE DOC] Public entry of the FSDP (fully sharded data parallel, FSDP2 style) training
# backend. Exports the Ray training actor class and the argument loader for this backend, via a
# guarded import: if the availability flag were false, both names would instead be replaced by a
# function that raises an import error with installation guidance when called (the flag is
# currently a constant true, see below). Also records two availability flags (memory saver and
# FSDP) as module constants.
# Side effect at import: sets the ROOT logger level to warning, which silences info and debug
# logging of every logger in the process that relies on inheritance.
# KNOWN ISSUE: both guarded blocks only assign a constant, so they can never raise an import
# error; the imports they were meant to test are missing, making the warnings and the fallback
# branch dead code, and the memory saver flag is never read.
import logging


try:
    _TORCH_MEMORY_SAVER_AVAILABLE = True
except ImportError:
    logging.warning("torch_memory_saver is not installed, refer to : https://github.com/fzyzcjy/torch_memory_saver")
    _TORCH_MEMORY_SAVER_AVAILABLE = False

try:
    _FSDP_AVAILABLE = True
except ImportError as e:
    logging.warning(f"FSDP backend dependencies not available: {e}")
    _FSDP_AVAILABLE = False

if _FSDP_AVAILABLE:
    from .actor import FSDPTrainRayActor
    from .arguments import load_fsdp_args
else:

    def _raise_import_error(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Stand-in for the unavailable FSDP backend entry points; accepts
        # any arguments and always raises an import error explaining that FSDP2 support is needed
        # and where to read about installing it. Currently unreachable (see package note).
        raise ImportError(
            "FSDP backend is not available. "
            "Please ensure PyTorch with FSDP2 support is installed. "
            "For installation instructions, refer to: https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html"
        )

    FSDPTrainRayActor = _raise_import_error
    load_fsdp_args = _raise_import_error

__all__ = ["load_fsdp_args", "FSDPTrainRayActor"]

logging.getLogger().setLevel(logging.WARNING)
