# [MODULE DOC] CPU tests of the value-model hook factory in the Megatron LoRA bridge helpers,
# which swaps a bridge-built model's language-model head for a scalar value head on the last
# pipeline stage. The Megatron surface the hook touches (a few modules and the parallel-state
# functions) is stubbed in the module table by a module-scoped fixture, and the real value-head class
# is replaced by a recording stand-in with the real keyword-only signature. Tests check the head
# is installed with the right sizes, receives the chunk's config, the factory takes only the hidden
# size, non-last virtual pipeline stages are skipped, and, by parsing the real class source, that
# the hook's constructor arguments bind to the real constructor signature.
"""Tests for the bridge value-model hook in lora/bridge.py.

``_make_value_model_hook`` swaps a bridge-built model's language-model head for
a scalar value head. Its body had no coverage, which let a constructor mismatch
go unnoticed: it passed ``sequence_parallel=`` to ``LinearForLastLayer``, whose
``__init__`` takes a keyword-only ``config`` and no ``sequence_parallel``. Any
run that reached the hook raised ``TypeError``. These tests exercise the hook
directly so the call site cannot drift from the constructor again.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])


import ast
import inspect
import sys
import types
from pathlib import Path

import pytest

_MISSING = object()


class _RecordingValueHead:
    # [CLASS DOC] HELPER: Stand-in for the real value-head layer with the same constructor shape
    # (input size, output size, keyword-only config, optional bias) that records its arguments, so
    # wrong keywords raise a type error like the real class.
    """Stand-in for ``LinearForLastLayer`` carrying its real signature.

    The keyword-only ``config`` is what makes this test meaningful: passing any
    other keyword, or omitting ``config``, raises ``TypeError`` exactly as the
    real class would.
    """

    def __init__(self, input_size: int, output_size: int, *, config, bias: bool = True) -> None:
        # [FUNCTION DOC] HELPER: Stores the four arguments as attributes.
        self.input_size = input_size
        self.output_size = output_size
        self.config = config
        self.bias = bias


class _FakeModelChunk:
    # [CLASS DOC] HELPER: Fake model chunk with a config object carrying a sequence-parallel flag
    # and an output layer initially set to the string "original-lm-head".
    def __init__(self, sequence_parallel: bool = False) -> None:
        # [FUNCTION DOC] HELPER: Sets the config (simple namespace with the flag) and the original
        # output layer marker.
        self.config = types.SimpleNamespace(sequence_parallel=sequence_parallel)
        self.output_layer = "original-lm-head"


@pytest.fixture(scope="module")
def helpers_module():
    # [FUNCTION DOC] HELPER: Module-scoped generator fixture. Installs stub modules in the module
    # table for the Megatron package, its core subpackage, tensor-parallel (with a dummy column-
    # parallel linear class), utils (with a no-op wrapped-model attribute getter) and a parallel-state
    # object; imports the LoRA bridge helpers module; and replaces the model-provider module with a stub
    # exposing the recording value-head class. Yields a namespace with the helpers module and the
    # parallel-state stub. Teardown (always) restores or removes every installed module table entry.
    # Done once per module because torch cannot be evicted and re-imported.
    """Stub the megatron surface the hook touches and import the helpers once.

    Module scope matters: torch cannot survive being evicted from sys.modules
    and re-imported, so the stubs are installed and reverted a single time.
    """
    installed: dict[str, object] = {}

    def install(name: str, module) -> None:
        # [FUNCTION DOC] HELPER: Remembers the module table's previous entry for the name (or a
        # missing marker) in the installed dictionary and then puts the module there.
        installed[name] = sys.modules.get(name, _MISSING)
        sys.modules[name] = module

    def stub(name: str, attrs: dict | None = None, is_package: bool = False):
        # [FUNCTION DOC] HELPER: Creates an empty module with the given name (with an empty search
        # path when it should act as a package), sets the given attributes, installs it, and returns it.
        module = types.ModuleType(name)
        if is_package:
            module.__path__ = []
        for attr_name, value in (attrs or {}).items():
            setattr(module, attr_name, value)
        install(name, module)
        return module

    parallel_state = types.SimpleNamespace()
    try:
        stub("megatron", is_package=True)
        stub("megatron.core", is_package=True)
        stub("megatron.core.tensor_parallel", {"ColumnParallelLinear": type("ColumnParallelLinear", (), {})})
        stub("megatron.core.utils", {"get_attr_wrapped_model": lambda *a, **k: None})
        install("megatron.core.parallel_state", parallel_state)

        from miles.backends.megatron_utils.lora import bridge as bridge_lora_helpers

        # The hook imports LinearForLastLayer lazily from the sibling module, so
        # swapping the module in sys.modules is enough to intercept it.
        stub(
            "miles.backends.megatron_utils.model_provider",
            {"LinearForLastLayer": _RecordingValueHead},
        )
        yield types.SimpleNamespace(helpers=bridge_lora_helpers, parallel_state=parallel_state)
    finally:
        for name, previous in installed.items():
            if previous is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


@pytest.fixture
def hook_env(helpers_module):
    # [FUNCTION DOC] HELPER: Per-test fixture that resets the stubbed parallel state to a single
    # pipeline stage with no virtual pipelining where every stage counts as the last stage, and returns
    # the shared helpers namespace.
    """Reset the stubbed topology to a single non-virtual pipeline stage."""
    helpers_module.parallel_state.get_pipeline_model_parallel_world_size = lambda: 1
    helpers_module.parallel_state.get_virtual_pipeline_model_parallel_world_size = lambda: None
    helpers_module.parallel_state.is_pipeline_last_stage = lambda **kwargs: True
    return helpers_module


def test_value_head_replaces_output_layer(hook_env):
    # [FUNCTION DOC] Builds the hook for hidden size 4096 and applies it to a list with one fake
    # chunk; asserts the chunk's output layer is now the recording value head with input size 4096 and
    # output size 1.
    """The hook installs a scalar value head on the last pipeline stage."""
    hook = hook_env.helpers._make_value_model_hook(4096)
    chunk = _FakeModelChunk()

    hook([chunk])

    assert isinstance(chunk.output_layer, _RecordingValueHead)
    assert chunk.output_layer.input_size == 4096
    assert chunk.output_layer.output_size == 1


def test_value_head_receives_the_chunk_config(hook_env):
    # [FUNCTION DOC] With a chunk whose config has sequence parallel on, asserts the installed head
    # received that very config object and its flag is true.
    """``config`` comes from the model chunk, which is where sequence_parallel lives."""
    hook = hook_env.helpers._make_value_model_hook(2048)
    chunk = _FakeModelChunk(sequence_parallel=True)

    hook([chunk])

    assert chunk.output_layer.config is chunk.config
    assert chunk.output_layer.config.sequence_parallel is True


def test_hook_takes_only_hidden_size(hook_env):
    # [FUNCTION DOC] Asserts the hook factory's parameter names are exactly the single name hidden
    # size.
    """The factory must not resurrect a redundant sequence_parallel parameter."""
    params = inspect.signature(hook_env.helpers._make_value_model_hook).parameters

    assert list(params) == ["hidden_size"]


def test_hook_skips_non_last_pipeline_stages(hook_env):
    # [FUNCTION DOC] Configures the stubbed topology with pipeline size 2, virtual pipeline size 2
    # and a last-stage predicate true only for virtual stage 1. Applies the hook to two chunks and
    # asserts the first keeps its original output layer and the second gets the value head.
    """With virtual pipelining, only last-stage chunks get a value head."""
    hook_env.parallel_state.get_pipeline_model_parallel_world_size = lambda: 2
    hook_env.parallel_state.get_virtual_pipeline_model_parallel_world_size = lambda: 2
    hook_env.parallel_state.is_pipeline_last_stage = lambda ignore_virtual, vp_stage: vp_stage == 1

    hook = hook_env.helpers._make_value_model_hook(1024)
    chunks = [_FakeModelChunk(), _FakeModelChunk()]

    hook(chunks)

    assert chunks[0].output_layer == "original-lm-head"
    assert isinstance(chunks[1].output_layer, _RecordingValueHead)


def test_value_head_kwargs_match_the_real_constructor(hook_env):
    # [FUNCTION DOC] Parses the real model-provider source file (found next to the megatron_utils
    # package) without importing Megatron, finds the real value-head class's constructor, and builds
    # a signature from it: its positional arguments after self become positional-or-keyword
    # parameters, and keyword-only arguments become keyword-only parameters whose defaults are literal-
    # evaluated (a missing default means required). Runs the hook on a fake chunk and asserts that the
    # arguments the hook passed (input size, output size, config) bind to that signature, i.e. the
    # call matches the real constructor. Raises a type error on mismatch.
    """Guard against drift: bind the hook's kwargs to the real LinearForLastLayer.

    The other tests use a stand-in class, so they would keep passing if the real
    constructor changed. This one parses the real signature out of
    ``model_provider.py`` and binds against it, with no megatron import needed.
    """
    import miles.backends.megatron_utils as megatron_utils_pkg

    source_path = Path(megatron_utils_pkg.__file__).parent / "model_provider.py"
    tree = ast.parse(source_path.read_text())
    init = next(
        node
        for cls in ast.walk(tree)
        if isinstance(cls, ast.ClassDef) and cls.name == "LinearForLastLayer"
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    parameters = [
        inspect.Parameter(arg.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD) for arg in init.args.args[1:]
    ] + [
        inspect.Parameter(
            arg.arg,
            inspect.Parameter.KEYWORD_ONLY,
            default=inspect.Parameter.empty if default is None else ast.literal_eval(default),
        )
        for arg, default in zip(init.args.kwonlyargs, init.args.kw_defaults, strict=True)
    ]

    hook = hook_env.helpers._make_value_model_hook(512)
    chunk = _FakeModelChunk()
    hook([chunk])
    head = chunk.output_layer

    inspect.Signature(parameters).bind(input_size=head.input_size, output_size=head.output_size, config=head.config)
