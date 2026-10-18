# [MODULE DOC]
# HELPER: Small ray helpers: an opaque box to keep values from being unwrapped by ray, and options
# that pin an actor or task to the head node.
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.util.state import list_nodes


class Box:
    # [CLASS DOC] HELPER: Read-only wrapper around a value (for example to pass an object reference
    # through ray without it being resolved automatically).
    def __init__(self, inner):
        # [FUNCTION DOC] HELPER: Stores the wrapped value.
        self._inner = inner

    @property
    def inner(self):
        # [FUNCTION DOC] HELPER: Property returning the wrapped value.
        return self._inner


def compute_ray_pin_head_options():
    # [FUNCTION DOC] HELPER: Returns a mapping with a hard node affinity scheduling strategy for the
    # head node, to pass as ray options. Queries the cluster state; raises a runtime error if no head
    # node is found.
    head_node_id = _get_head_node_id()
    return {
        "scheduling_strategy": NodeAffinitySchedulingStrategy(
            node_id=head_node_id,
            soft=False,
        )
    }


def _get_head_node_id() -> str:
    # [FUNCTION DOC] HELPER: Id of the first listed node flagged as head; raises a runtime error if
    # none. Uses the ray state api (needs a running cluster).
    for node in list_nodes():
        if node.is_head_node:
            return node.node_id
    raise RuntimeError("Could not find a head node in the Ray cluster")
