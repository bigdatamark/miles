# [MODULE DOC] Test that the shared agentic session and credentials modules can be imported in a
# fresh interpreter without pulling in the torch library.
import subprocess
import sys


def test_shared_package_is_torch_free():
    # [FUNCTION DOC] Spawns a fresh Python interpreter that imports the agentic session and
    # credentials modules and exits non-zero if torch is among the loaded modules. Asserts the
    # child's exit code is zero.
    """Agent functions import it on CPU-only hosts and in offline tests (see nemogym_agent_function)."""
    code = (
        "import sys; import miles.rollout.agentic.session, miles.rollout.agentic.credentials; "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
