# [PACKAGE DOC] Command-line layer of the run_megatron tool. Re-exports the typer application
# object built in the commands subpackage; also holds helper modules for parallelism configuration
# (parallel_utils), megatron path resolution (path_utils), prompt token generation
# (prompt_utils) and torchrun command construction (worker_executor).
from miles.utils.debug_utils.run_megatron.cli.commands import app

__all__ = ["app"]
