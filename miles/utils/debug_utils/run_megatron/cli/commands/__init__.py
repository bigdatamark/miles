# [PACKAGE DOC] Defines the shared typer application (with pretty exception rendering turned off,
# so errors show as plain tracebacks) and registers the commands from the run, compare and
# run_and_compare modules on it. The args module holds the dataclasses describing each command's
# options. The module docstring below mentions a show-model-args command, which is registered by
# the run module.
"""CLI commands for run_megatron.

Usage:
    python -m miles.utils.debug_utils.run_megatron run ...
    python -m miles.utils.debug_utils.run_megatron compare ...
    python -m miles.utils.debug_utils.run_megatron run-and-compare ...
    python -m miles.utils.debug_utils.run_megatron show-model-args ...
"""

import typer

from miles.utils.debug_utils.run_megatron.cli.commands import compare, run, run_and_compare

app: typer.Typer = typer.Typer(pretty_exceptions_enable=False)

run.register(app)
compare.register(app)
run_and_compare.register(app)
