# [MODULE DOC] Static source-scan tests (parsing the Megatron actor module's syntax tree, never
# importing it) that pin the structure of the trainer's state loading: the model-state load function
# is called from exactly one method (the reusable core loader); init reaches the core loader; the
# weight updater is built only in one helper called once from the training-state initializer, after
# the state load and before the offload sleep in init; and the core loader never calls sleep. Source
# location comes from the shared source-scan helper's framework root.
import ast

from tests.fast.source_scan import FRAMEWORK_ROOT

# reading the source, not the module: importing it pulls in torch_memory_saver, which a cpu shard
# does not have, and a collection error there takes down every other test in the same shard
_SOURCE = FRAMEWORK_ROOT / "backends" / "megatron_utils" / "actor.py"
_CORE_METHOD = "_load_state_core"
_LOAD_FUNCTION = "load_model_state"
_INIT_METHOD = "init"
_TRAINING_STATE_METHOD = "_init_training_state"
_WEIGHT_UPDATER_METHOD = "_init_weight_updater_and_publisher"
_SLEEP_METHOD = "sleep"
_WEIGHT_UPDATER_ATTRIBUTE = "weight_updater"


def _functions_calling(name: str, *, as_method: bool) -> list[str]:
    # [FUNCTION DOC] HELPER: Parses the actor source and returns the sorted, de-duplicated names of
    # all (sync or async) function definitions anywhere in the file whose body contains a call to the
    # given name. With as_method true only attribute-style calls (receiver.name(...)) match by attribute
    # name; otherwise only plain-name calls match. Note nested functions' calls count for every
    # enclosing function too, since the walk descends into them.
    tree = ast.parse(_SOURCE.read_text())
    calling: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call):
                continue
            called = inner.func.attr if as_method and isinstance(inner.func, ast.Attribute) else None
            if not as_method and isinstance(inner.func, ast.Name):
                called = inner.func.id
            if called == name:
                calling.append(node.name)
    return sorted(set(calling))


class TestWhereTheTrainerLoadsItsState:
    # [CLASS DOC] Tests of the call sites of the trainer's state loading.
    def test_the_state_load_is_reached_from_one_function_only(self):
        # [FUNCTION DOC] Asserts the only function calling the model-state load function by plain name
        # is the core loader method.
        """A second call site would let a reload drift from what init does, which is what this op exists to stop."""
        assert _functions_calling(_LOAD_FUNCTION, as_method=False) == [_CORE_METHOD]

    def test_init_reaches_its_state_load_through_that_function(self):
        # [FUNCTION DOC] Asserts the init method is among the functions that call the core loader as
        # a method.
        """The invariant is worth nothing if init stopped going through the function everything else goes through."""
        assert "init" in _functions_calling(_CORE_METHOD, as_method=True)


def _function(name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    # [FUNCTION DOC] HELPER: Parses the actor source and returns the single function definition with
    # the given name anywhere in the tree; raises a value error (from tuple unpacking) if there are
    # zero or several.
    tree = ast.parse(_SOURCE.read_text())
    (found,) = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    return found


def _method_call_lines(function_name: str, method: str) -> list[int]:
    # [FUNCTION DOC] HELPER: Returns the sorted source line numbers of attribute-style calls
    # (receiver.method(...)) of the given method name inside the named function (including nested code).
    return sorted(
        inner.lineno
        for inner in ast.walk(_function(function_name))
        if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute) and inner.func.attr == method
    )


def _attribute_assignment_lines(function_name: str, attribute: str) -> list[int]:
    # [FUNCTION DOC] HELPER: Returns the sorted line numbers of plain assignment statements inside
    # the named function whose target is an attribute with the given name (augmented and annotated
    # assignments are not counted).
    return sorted(
        target.lineno
        for node in ast.walk(_function(function_name))
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute) and target.attr == attribute
    )


class TestWhenTheTrainerBuildsItsWeightUpdater:
    # [CLASS DOC] Tests of where and when the weight updater is built during init.
    def test_the_weight_updater_is_built_where_init_says_it_is(self):
        # [FUNCTION DOC] Asserts: init itself never assigns the weight-updater attribute; the only
        # function calling the updater-building helper is the training-state initializer, which calls it
        # exactly once; and the helper assigns the weight-updater attribute exactly twice.
        """The ordering below reads init, so the build has to stay in the one call init makes for it."""
        assert _attribute_assignment_lines(_INIT_METHOD, _WEIGHT_UPDATER_ATTRIBUTE) == []
        assert _functions_calling(_WEIGHT_UPDATER_METHOD, as_method=True) == [_TRAINING_STATE_METHOD]
        assert len(_method_call_lines(_TRAINING_STATE_METHOD, _WEIGHT_UPDATER_METHOD)) == 1
        assert len(_attribute_assignment_lines(_WEIGHT_UPDATER_METHOD, _WEIGHT_UPDATER_ATTRIBUTE)) == 2

    def test_the_weight_updater_is_built_after_the_state_load(self):
        # [FUNCTION DOC] Finds the single line where init calls the training-state initializer and
        # asserts the last line where init calls the core loader is before it.
        """The checkpoint load lifts expert_bias back to fp32, so a snapshot taken before it records a stale dtype."""
        (built,) = _method_call_lines(_INIT_METHOD, _TRAINING_STATE_METHOD)

        assert max(_method_call_lines(_INIT_METHOD, _CORE_METHOD)) < built

    def test_the_trainer_only_offloads_once_the_weight_updater_exists(self):
        # [FUNCTION DOC] Asserts the last line where init calls sleep is after the single line where
        # init calls the training-state initializer.
        """Building it needs a live cuda context and live process groups, which the offload sleep tears down."""
        (built,) = _method_call_lines(_INIT_METHOD, _TRAINING_STATE_METHOD)

        assert max(_method_call_lines(_INIT_METHOD, _SLEEP_METHOD)) > built

    def test_the_reusable_load_never_offloads_by_itself(self):
        # [FUNCTION DOC] Asserts the core loader contains no call to sleep.
        """A reload runs it with the trainer awake, and an offload there would strand the caller asleep."""
        assert _method_call_lines(_CORE_METHOD, _SLEEP_METHOD) == []
