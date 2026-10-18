# [MODULE DOC]
# HELPER: Observes a Kubernetes release (a deployed training run) while hot restarts replace
# parts of it, and records what was seen as immutable snapshots. A snapshot holds facts about
# every pod (name, uid, summed container restart count), every workload of the two supported
# kinds (stateful sets and leader-worker sets: name, generation, a SHA-256 fingerprint of the
# pod templates, and the restart-stamp annotation of the templates), the trainer rpc server's
# boot uuid (read from a response header of its health path), and which kinds of read failed.
# A background polling observer takes snapshots at an interval, deciding which reads are usable
# (complete, not a release being installed or removed, topology settled, no workload missing)
# and counting attempts and failures. Also provides pure parsing functions over the JSON
# listings of kubectl and name computations. Cluster reads go through a kubectl helper; the
# trainer is reached over HTTP at its in-cluster service address.
import hashlib
import json
import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import requests

from miles.ray.specs.rollout import ROLLOUT_EXECUTOR_POOL_ID
from miles.ray.specs.train import compute_trainer_controller_pool_id
from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import RESTART_AT_ANNOTATION
from miles.utils.external_utils.command_utils.helm_backend.naming import ORCHESTRATOR_COMPONENT, RunNames
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.test_utils.kubectl_reads import read_objects_of_release
from miles.utils.test_utils.polling_worker import PollingWorker, poll_until_stopped
from miles.utils.workers.rpc.common.protocol import BOOT_UUID_HEADER, HEALTH_PATH
from miles.utils.workers.worker_provider.kubernetes.helm.naming import component_name, static_worker_host
from miles.utils.workers.worker_spec import DEFAULT_RPC_PORT


# ================================= constants ==================================


logger = logging.getLogger(__name__)

POD_KIND: str = "pods"
STATEFUL_SET_KIND: str = "statefulsets"
LEADER_WORKER_SET_KIND: str = "leaderworkersets.leaderworkerset.x-k8s.io"
WORKLOAD_KINDS: tuple[str, ...] = (STATEFUL_SET_KIND, LEADER_WORKER_SET_KIND)
BOOT_UUID_TIMEOUT_SECONDS: float = 10.0
POLL_INTERVAL_SECONDS: float = 5.0
JOIN_TIMEOUT_SECONDS: float = 1800.0


# ============================ what a cluster holds ============================


class PodFact(FrozenStrictBaseModel):
    # [CLASS DOC] HELPER: Immutable facts about one pod: its name, its uid (changes when the pod is
    # recreated) and the sum of restart counts of all its containers.
    name: str
    uid: str
    restart_count: int


class WorkloadFact(FrozenStrictBaseModel):
    # [CLASS DOC] HELPER: Immutable facts about one workload object: its kind, name, generation
    # number, a fingerprint of its pod template(s), and the restart stamp annotation value found
    # on its pod templates (missing when it was never stamped).
    kind: str
    name: str
    generation: int
    pod_template_fingerprint: str
    restart_at: str | None


class ClusterSnapshot(FrozenStrictBaseModel):
    # [CLASS DOC] HELPER: Immutable snapshot of a release at one moment: the pod facts, the
    # workload facts, the trainer rpc server boot uuid (missing if it could not be reached) and
    # the names of the object kinds whose read failed (empty by default).
    pods: tuple[PodFact, ...]
    workloads: tuple[WorkloadFact, ...]
    trainer_boot_uuid: str | None
    reads_missing: tuple[str, ...] = ()

    @property
    def workload_names(self) -> tuple[str, ...]:
        # [FUNCTION DOC] HELPER: Names of the workloads in this snapshot, in stored order.
        return tuple(one.name for one in self.workloads)

    @property
    def describes_whole_release(self) -> bool:
        # [FUNCTION DOC] HELPER: True when none of the three read kinds (pods and the two
        # workload kinds) is listed as having failed to read. Note the trainer uuid is not
        # part of this.
        return not ({POD_KIND, *WORKLOAD_KINDS} & set(self.reads_missing))

    @property
    def describes_gone_release(self) -> bool:
        # [FUNCTION DOC] HELPER: True when there are no pods or no workloads, i.e. the release is
        # being installed or uninstalled.
        return not self.pods or not self.workloads


# ============================== observing a run ===============================


@dataclass
class ClusterObserver:
    # [CLASS DOC] HELPER: Mutable collector of cluster snapshots for one release. Fields: the
    # release name, namespace and trainer id (used to find the trainer rpc server); the list of
    # accepted snapshots; the number of observation attempts (counted only once the release has
    # been seen up, plus accepted ones) and of failed reads; whether the release was ever seen
    # up. Private state: the workload names the release settled into (set when two consecutive
    # reads show the same workloads and pods) and the last topology read before settling.
    # Not thread safe on its own; used from one polling thread, then once more after it stops.
    release: str
    namespace: str
    trainer_id: str
    snapshots: list[ClusterSnapshot] = field(default_factory=list)
    attempts: int = 0
    failures: int = 0
    release_seen_up: bool = False
    _settled_workloads: frozenset[str] | None = field(default=None, init=False)
    _topology_read_before: tuple[frozenset[str], frozenset[str]] | None = field(default=None, init=False)

    def observe_once_or_warn(self) -> None:
        # [FUNCTION DOC] Run one observation; any exception at all (including interrupts, since the
        # broadest base exception class is caught) is recorded as a failed read and logged as a
        # warning with its traceback, and swallowed.
        try:
            self.observe_once()
        except BaseException:
            self._record_failed_read()
            logger.warning("Failed to observe the cluster of a run being hot restarted", exc_info=True)

    def observe_once(self) -> None:
        # [FUNCTION DOC] Take one snapshot of the release (querying the cluster and the trainer rpc
        # server) and decide whether to keep it. A snapshot missing any pod or workload read is
        # counted as a failed read and dropped. A snapshot with no pods or no workloads
        # (release being installed or removed) is dropped with a warning and not counted. A
        # snapshot of a release not yet settled, or lacking a previously settled workload (partial
        # listing), is dropped with a warning and not counted. Otherwise the release is marked seen
        # up, the attempt count is incremented and the snapshot appended. Exceptions from
        # reading propagate.
        snapshot = read_cluster_snapshot(
            release=self.release,
            namespace=self.namespace,
            trainer_rpc_url=compute_trainer_rpc_url(
                release=self.release, namespace=self.namespace, trainer_id=self.trainer_id
            ),
        )
        if not snapshot.describes_whole_release:
            self._record_failed_read()
            logger.warning(
                f"Observing {self.release} could not read {list(snapshot.reads_missing)}, so this is a read that "
                f"failed rather than a run whose pods changed"
            )
            return
        if snapshot.describes_gone_release:
            logger.warning(
                f"Observed {self.release} with {len(snapshot.pods)} pod(s) and {len(snapshot.workloads)} "
                f"workload(s), which is a release being installed or uninstalled rather than a run whose pods "
                f"were replaced"
            )
            return
        if not self._describes_settled_release(snapshot):
            return

        self.release_seen_up = True
        self.attempts += 1
        self.snapshots.append(snapshot)

    def _describes_settled_release(self, snapshot: ClusterSnapshot) -> bool:
        # [FUNCTION DOC] HELPER: Decide whether a snapshot may be recorded. Before settling: if the
        # (workload names, pod names) pair differs from the one remembered from the previous read,
        # remember it, log a warning and return false; if it equals the remembered pair, store the
        # workload names as settled, log it, and return true. After settling: if any settled
        # workload is missing from the snapshot, log a warning and return false; else true (new
        # workloads appearing is allowed).
        workload_names = frozenset(snapshot.workload_names)
        pod_names = frozenset(one.name for one in snapshot.pods)
        if self._settled_workloads is None:
            if (workload_names, pod_names) != self._topology_read_before:
                self._topology_read_before = (workload_names, pod_names)
                logger.warning(
                    f"Observed {self.release} holding {sorted(workload_names)} over {sorted(pod_names)}, which no "
                    f"read before it had seen, so the release is still being installed rather than being watched"
                )
                return False
            self._settled_workloads = workload_names
            logger.info(f"{self.release} settled into {sorted(self._settled_workloads)}; recording from here on")
            return True

        if missing := self._settled_workloads - workload_names:
            logger.warning(
                f"Observing {self.release} listed neither {sorted(missing)} nor a release that is gone, so this is "
                f"a partial listing rather than a run whose workloads were removed"
            )
            return False
        return True

    def _record_failed_read(self) -> None:
        # [FUNCTION DOC] HELPER: Count a failed read; if the release has been seen up, also count it
        # as an observation attempt (so failures before the release came up do not lower the
        # success ratio).
        self.failures += 1
        if self.release_seen_up:
            self.attempts += 1


@contextmanager
def observing_cluster(
    observer: ClusterObserver, *, poll_interval_seconds: float = POLL_INTERVAL_SECONDS
) -> Iterator[ClusterObserver]:
    # [FUNCTION DOC] Context manager that polls the observer in a background worker thread (named
    # "hot-restart-observer") every poll interval (default 5 seconds) for the duration of the
    # with-block, yielding the observer. On exit (even after an error) it stops the worker and
    # waits up to 1800 seconds. After a clean stop it asserts the worker is no longer running
    # (otherwise reading the results would race it), then takes one final observation with
    # exceptions propagating. Note the final observation is skipped if the body raised.
    def observe_until_stopped(stop_event: threading.Event) -> None:
        # [FUNCTION DOC] HELPER: Worker body: poll the observer's warn-only observation until the
        # stop event is set, at the poll interval.
        poll_until_stopped(stop_event, tick=observer.observe_once_or_warn, poll_interval_seconds=poll_interval_seconds)

    worker = PollingWorker(name="hot-restart-observer", run=observe_until_stopped)
    worker.start()
    try:
        yield observer
    finally:
        worker.stop_and_join(timeout_seconds=JOIN_TIMEOUT_SECONDS)

    worker.assert_not_running(
        message=(
            f"the cluster observer was still reading the run {JOIN_TIMEOUT_SECONDS}s after being asked to stop, "
            f"so reading what it collected now would race it"
        )
    )
    observer.observe_once()


# ============================== reading a cluster =============================


def compute_trainer_rpc_url(*, release: str, namespace: str, trainer_id: str) -> str:
    # [FUNCTION DOC] Build the HTTP URL of the trainer rpc server's health path: the in-cluster
    # service host name of the static worker (index 0) of the trainer controller pool for the
    # given trainer id in the release and namespace, plus the default rpc port and the health path.
    host = RunNames.service_fqdn(
        name=static_worker_host(release, compute_trainer_controller_pool_id(trainer_id), 0), namespace=namespace
    )
    return f"http://{host}:{DEFAULT_RPC_PORT}{HEALTH_PATH}"


def read_cluster_snapshot(*, release: str, namespace: str, trainer_rpc_url: str) -> ClusterSnapshot:
    # [FUNCTION DOC] Build one snapshot. Reads the boot uuid from the trainer URL (missing on any
    # failure), then lists, as JSON, the pods and the two workload kinds of the release in the
    # namespace (each listing may fail, giving nothing). Pod facts come from the pod listing
    # (empty if it failed); workload facts from the available workload listings, sorted by kind
    # then name. The names of kinds whose listing failed go into the missing-reads field. Side
    # effects: network and cluster command calls.
    boot_uuid = read_boot_uuid(trainer_rpc_url)
    payload_of_kind = {
        kind: _read_objects(kind=kind, release=release, namespace=namespace) for kind in (POD_KIND, *WORKLOAD_KINDS)
    }

    pods = payload_of_kind[POD_KIND]
    workloads = tuple(
        fact
        for kind in WORKLOAD_KINDS
        if (payload := payload_of_kind[kind]) is not None
        for fact in parse_workload_facts(payload, kind=kind)
    )
    return ClusterSnapshot(
        pods=parse_pod_facts(pods) if pods is not None else (),
        workloads=tuple(sorted(workloads, key=lambda one: (one.kind, one.name))),
        trainer_boot_uuid=boot_uuid,
        reads_missing=tuple(kind for kind, payload in payload_of_kind.items() if payload is None),
    )


def compute_hot_restart_workloads(release: str) -> frozenset[str]:
    # [FUNCTION DOC] Return the frozen set of the two workload names a hot restart rolls: the
    # chart-rendered names of the orchestrator component and of the rollout executor pool in the
    # release.
    return frozenset(
        component_name(release, component) for component in (ORCHESTRATOR_COMPONENT, ROLLOUT_EXECUTOR_POOL_ID)
    )


def read_restart_stamp_of_workload(*, release: str, namespace: str) -> dict[str, str | None] | None:
    # [FUNCTION DOC] List the two workload kinds of the release and return a mapping from workload
    # name to its restart stamp (missing if unstamped). Returns nothing if either listing fails.
    # Side effect: cluster command calls.
    stamps: dict[str, str | None] = {}
    for kind in WORKLOAD_KINDS:
        payload = _read_objects(kind=kind, release=release, namespace=namespace)
        if payload is None:
            return None
        stamps.update({fact.name: fact.restart_at for fact in parse_workload_facts(payload, kind=kind)})
    return stamps


def read_boot_uuid(url: str) -> str | None:
    # [FUNCTION DOC] HTTP GET of the URL with a 10 second timeout; raise-for-status; return the
    # boot uuid response header value (missing if the header is absent). Any exception is logged
    # as a warning and turned into a missing result.
    try:
        response = requests.get(url, timeout=BOOT_UUID_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.headers.get(BOOT_UUID_HEADER)
    except Exception:
        logger.warning(f"Failed to read the rpc server boot uuid from {url}", exc_info=True)
        return None


def parse_pod_facts(payload: dict) -> tuple[PodFact, ...]:
    # [FUNCTION DOC] Convert a pod listing payload (mapping with an items list) into pod facts:
    # name and uid from metadata, restart count as the sum over all container statuses of their
    # restart count (zero when status or count is absent). Returns facts sorted by name. A missing
    # items, name or uid key raises a key error.
    facts = [
        PodFact(
            name=item["metadata"]["name"],
            uid=item["metadata"]["uid"],
            restart_count=sum(
                int(one.get("restartCount", 0)) for one in item.get("status", {}).get("containerStatuses", [])
            ),
        )
        for item in payload["items"]
    ]
    return tuple(sorted(facts, key=lambda one: one.name))


def parse_workload_facts(payload: dict, *, kind: str) -> tuple[WorkloadFact, ...]:
    # [FUNCTION DOC] Convert a workload listing payload into facts of the given kind: name and
    # integer generation from metadata, the pod template fingerprint and the restart stamp (both
    # computed from the object's pod templates; they raise an assertion error if the object has
    # no template, or if its templates carry more than one distinct stamp). Returns facts sorted
    # by name.
    facts = [
        WorkloadFact(
            kind=kind,
            name=item["metadata"]["name"],
            generation=int(item["metadata"]["generation"]),
            pod_template_fingerprint=_compute_pod_template_fingerprint(item),
            restart_at=_read_restart_at(item),
        )
        for item in payload["items"]
    ]
    return tuple(sorted(facts, key=lambda one: one.name))


def _read_objects(*, kind: str, release: str, namespace: str) -> dict | None:
    # [FUNCTION DOC] HELPER: List the objects of the kind belonging to the release in the namespace
    # as JSON via the cluster read helper and parse it. On any exception, log a warning and
    # return nothing.
    try:
        return json.loads(read_objects_of_release(kind=kind, release=release, namespace=namespace, output="json"))
    except Exception:
        logger.warning(f"Failed to list the {kind} of {release} in {namespace}", exc_info=True)
        return None


def _read_restart_at(item: dict) -> str | None:
    # [FUNCTION DOC] HELPER: Collect the restart-stamp annotation values from all pod templates of
    # the workload item. Asserts there is at most one distinct value (a hot restart writes one
    # stamp per object). Returns that value or nothing.
    templates = _read_pod_templates(item)
    stamps = {
        stamp
        for template in templates
        if (stamp := template.get("metadata", {}).get("annotations", {}).get(RESTART_AT_ANNOTATION)) is not None
    }
    assert len(stamps) <= 1, (
        f"{item['metadata']['name']} carries the restart stamps {sorted(stamps)} on the templates of one object, "
        f"and a hot restart writes one stamp per object"
    )
    return next(iter(stamps), None)


def _compute_pod_template_fingerprint(item: dict) -> str:
    # [FUNCTION DOC] HELPER: Serialize the item's pod templates as JSON with sorted keys and
    # compact separators and return the hexadecimal SHA-256 digest of that text. Note the
    # restart stamp annotation is inside the templates, so stamping also changes the fingerprint.
    text = json.dumps(_read_pod_templates(item), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def _read_pod_templates(item: dict) -> tuple[dict, ...]:
    # [FUNCTION DOC] HELPER: Return the pod templates present in the workload item: the ordinary
    # template of its spec (stateful set) and the leader template and worker template of its
    # leader-worker template (leader-worker set), omitting missing ones. Asserts at least one is
    # present.
    spec = item.get("spec", {})
    leader_worker_template = spec.get("leaderWorkerTemplate", {})
    templates = [
        spec.get("template"),
        leader_worker_template.get("leaderTemplate"),
        leader_worker_template.get("workerTemplate"),
    ]
    present = tuple(template for template in templates if template is not None)
    assert present, (
        f"{item['metadata']['name']} has no StatefulSet or LeaderWorkerSet pod template, so a hot restart cannot "
        f"tell whether that workload was rewritten"
    )
    return present
