# [MODULE DOC] Minimal environment check that torch's fully sharded data parallel wrapper can
# be imported.
import pytest


def test_fsdp_import():
    # [FUNCTION DOC] Tries to import the fully sharded data parallel class from torch's
    # distributed package; skips the test if the import fails with an import error, otherwise
    # asserts the class is not none.
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except ImportError:
        pytest.skip("FSDP not available in this environment")
    assert FSDP is not None
