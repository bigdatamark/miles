# [MODULE DOC] Pod-level helpers of the pairing controller: the coordinate model identifying a
# pod by pool, cell and pod index (with a slash-joined key), reading that coordinate from a pod's
# labels, reading and testing the pairing scheduling gate, and building the JSON patch that
# releases a gated inference pod onto a chosen node.
from __future__ import annotations

from collections.abc import Mapping

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.k8s_types import Pod
from miles.utils.workers.worker_provider.kubernetes.core import pod_view
from miles.utils.workers.worker_provider.kubernetes.helm.env import DEFAULT_LABEL_KEYS

_GATE_NAME = "miles.radixark.io/colocate-pairing"

_NODE_NAME_FIELD = "metadata.name"


class PodCoordinate(FrozenStrictBaseModel):
    # [CLASS DOC] Immutable, hashable pod identity: pool id, cell index and pod-in-cell index.
    pool_id: str
    cell_index: int
    pod_in_cell_index: int

    @property
    def key(self) -> str:
        # [FUNCTION DOC] Property: "pool/cell/pod" as text.
        return f"{self.pool_id}/{self.cell_index}/{self.pod_in_cell_index}"


def coordinate_of(pod: Pod) -> PodCoordinate | None:
    # [FUNCTION DOC] Parses the pod with the shared pod-view parser and default label keys;
    # returns none if the pod is not a recognized worker pod, else its coordinate.
    parsed = pod_view.parse_pod(pod, DEFAULT_LABEL_KEYS)
    if parsed is None:
        return None
    return PodCoordinate(
        pool_id=parsed.pool_id, cell_index=parsed.cell_index, pod_in_cell_index=parsed.pod_in_cell_index
    )


def release_patch(
    *,
    node_name: str,
    base_gpu_id: int,
    gates: list[str],
    annotations: Mapping[str, str],
) -> list[dict[str, object]]:
    # [FUNCTION DOC] Keyword-only. Builds a JSON-patch operation list for a gated pod. Asserts the
    # pod has annotations (adding into a missing map would be refused by the API server). Finds
    # the index of the pairing gate in the given gate names (a value error if absent). Operations,
    # in order: (1) add a required node affinity whose single selector term matches the node's
    # name field equal to the given node; (2) add the base-GPU-id annotation (key from the default
    # label keys, with slash and tilde escaped for JSON pointers) set to the id as text; (3) test
    # that the gate at that index still has the pairing gate's name (making the patch fail if
    # gates changed concurrently); (4) remove the gate at that index.
    # KNOWN ISSUE: the affinity operation adds the whole affinity field, replacing any affinity
    # the pod already had.
    key = DEFAULT_LABEL_KEYS.base_gpu_id_annotation
    assert annotations, (
        f"the pod carries no annotations, so adding {key} under a map that does not exist is a patch "
        f"the apiserver refuses"
    )

    index = gates.index(_GATE_NAME)
    pin = {
        "op": "add",
        "path": "/spec/affinity",
        "value": {
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [
                        {"matchFields": [{"key": _NODE_NAME_FIELD, "operator": "In", "values": [node_name]}]}
                    ]
                }
            }
        },
    }
    return [
        pin,
        {"op": "add", "path": f"/metadata/annotations/{_escape_pointer(key)}", "value": str(base_gpu_id)},
        {"op": "test", "path": f"/spec/schedulingGates/{index}/name", "value": _GATE_NAME},
        {"op": "remove", "path": f"/spec/schedulingGates/{index}"},
    ]


def gate_names(pod: Pod) -> list[str]:
    # [FUNCTION DOC] Names of the pod's scheduling gates in order.
    # KNOWN ISSUE: Kubernetes clients report the gate list as none for a pod without gates, in
    # which case iterating it raises a type error; the controller calls this on pods that may be
    # ungated.
    return [gate.name for gate in pod.spec.scheduling_gates]


def is_gated(pod: Pod) -> bool:
    # [FUNCTION DOC] True when the pod still has the pairing scheduling gate.
    return _GATE_NAME in gate_names(pod)


def _escape_pointer(token: str) -> str:
    # [FUNCTION DOC] HELPER: Escapes a JSON-pointer segment: tilde becomes tilde-zero first, then
    # slash becomes tilde-one.
    return token.replace("~", "~0").replace("/", "~1")
