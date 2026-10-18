# [MODULE DOC] Tests for the pure cell view functions of the kubernetes provider: compute_cell_info
# (aliveness requires every expected pod index present exactly once, ready and not being deleted;
# unknown cell size judged on readiness alone; members hash changes on deletion; cell metadata
# merged from the pods, conflicting values raise an assertion error) and compute_worker_infos (one
# worker info per worker a pod serves, with gpu ids split per worker, rpc ports offset per worker,
# bracketed ipv6 hosts, no worker class for command pods, and assertion errors for missing ports,
# missing addresses or missing pods). Pods are built with make_parsed_pod, runs with make_run.
from __future__ import annotations

import pytest
from tests.fast.utils.workers.worker_provider.kubernetes.run_specs import make_pool_spec

from miles.utils.workers.naming import compute_worker_name
from miles.utils.workers.worker_provider.kubernetes.core.cell_view import compute_cell_info, compute_worker_infos
from miles.utils.workers.worker_provider.kubernetes.core.pod_view import CellLabelKeys, ParsedPod
from miles.utils.workers.worker_provider.kubernetes.core.provider import KubernetesRunInfo

CELL_ID = "engine-00000"
ENGINE_CLASS = f"{__name__}.FakeEngine"


class FakeEngine:
    # [CLASS DOC] HELPER: Placeholder engine class referenced by import path as the pool's worker
    # class.
    def generate(self, prompt: str) -> str:
        # [FUNCTION DOC] HELPER: Returns the prompt unchanged.
        return prompt


def make_parsed_pod(
    *, pod_in_cell_index: int, cell_size: int = 2, ready: bool = True, deleting: bool = False, **kwargs
) -> ParsedPod:
    # [FUNCTION DOC] HELPER: Builds a parsed pod of cell engine-00000 (pool engine, cell index 0)
    # with the given pod index, expected cell size (default 2), readiness and deletion flags. The
    # pod name is engine-0-<index>, uid uid-<index>, restart count 0. Extra keyword arguments are
    # consumed: pod_ip (default 10.0.0.<index+1>), subdomain (none), gpu_ids (empty tuple), meta
    # (empty mapping); any other keyword is silently ignored.
    pod_ip = kwargs.pop("pod_ip", f"10.0.0.{pod_in_cell_index + 1}")
    return ParsedPod(
        name=f"engine-0-{pod_in_cell_index}",
        cell_id=CELL_ID,
        cell_index=0,
        pool_id="engine",
        pod_in_cell_index=pod_in_cell_index,
        ready=ready,
        deleting=deleting,
        pod_ip=pod_ip,
        uid=f"uid-{pod_in_cell_index}",
        restart_count=0,
        cell_size=cell_size,
        subdomain=kwargs.pop("subdomain", None),
        gpu_ids=kwargs.pop("gpu_ids", ()),
        meta=kwargs.pop("meta", {}),
    )


def make_run(
    *,
    workers_per_pod: int = 1,
    worker_class: str | None = ENGINE_CLASS,
    ports: dict[str, int] | None = None,
) -> KubernetesRunInfo:
    # [FUNCTION DOC] HELPER: Builds run info for namespace rl with one pool spec named engine
    # (ports default to rpc 8000 when none given; an explicit empty mapping means no ports; worker
    # class default is the fake engine; workers per pod default 1) and label keys pool-id,
    # cell-index, pod-index, cell-size, meta- prefix, gpu_ids and base-gpu-id.
    return KubernetesRunInfo(
        namespace="rl",
        label_selector="app.kubernetes.io/instance=r",
        specs={
            "engine": make_pool_spec(
                "engine",
                ports={"rpc": 8000} if ports is None else ports,
                worker_class=worker_class,
                workers_per_pod=workers_per_pod,
            )
        },
        label_keys=CellLabelKeys(
            pool_id="pool-id",
            cell_index="cell-index",
            pod_in_cell_index="pod-index",
            cell_size_annotation="cell-size",
            meta_annotation_prefix="meta-",
            gpu_ids_meta="gpu_ids",
            base_gpu_id_annotation="base-gpu-id",
        ),
    )


def build_cell_info(pods: list[ParsedPod]):
    # [FUNCTION DOC] HELPER: Computes the cell info of the engine cell from the pods with the
    # default run.
    return compute_cell_info(CELL_ID, pods=pods, run=make_run())


def build_worker_infos(pods: list[ParsedPod], *, workers_per_pod: int = 1):
    # [FUNCTION DOC] HELPER: Computes the worker infos of the engine cell from the pods with a run
    # whose pool has the given workers per pod (default 1).
    return compute_worker_infos(CELL_ID, pods=pods, run=make_run(workers_per_pod=workers_per_pod))


class TestCellLiveness:
    # [CLASS DOC] Tests of the aliveness flag of the cell info.
    def test_a_cell_whose_pods_all_arrived_and_are_ready_is_alive(self):
        # [FUNCTION DOC] Pods 0 and 1 of an expected 2, both ready: alive.
        """This is the only state in which a consumer may hand the cell work."""
        assert (
            build_cell_info([make_parsed_pod(pod_in_cell_index=0), make_parsed_pod(pod_in_cell_index=1)]).alive is True
        )

    def test_a_cell_still_missing_a_pod_is_not_alive(self):
        # [FUNCTION DOC] Only pod 0 of an expected 2: not alive.
        """A group being created has ready pods long before it has all of them."""
        assert build_cell_info([make_parsed_pod(pod_in_cell_index=0)]).alive is False

    def test_a_cell_that_shows_the_same_pod_index_twice_is_not_alive(self):
        # [FUNCTION DOC] Two pods both with index 0 (count matches size 2 but indices do not cover
        # 0 and 1): not alive.
        """Mid-replacement the old pod's deletion may lag its replacement, and a count would call that complete."""
        pods = [make_parsed_pod(pod_in_cell_index=0), make_parsed_pod(pod_in_cell_index=0)]

        assert build_cell_info(pods).alive is False

    def test_a_cell_whose_platform_publishes_no_size_is_judged_on_readiness_alone(self):
        # [FUNCTION DOC] One ready pod with cell size 0 (unknown): alive.
        """A platform that does not say how big a cell is cannot be second-guessed."""
        assert build_cell_info([make_parsed_pod(pod_in_cell_index=0, cell_size=0)]).alive is True

    def test_a_cell_with_an_unready_pod_is_not_alive(self):
        # [FUNCTION DOC] Both pods present but pod 1 not ready: not alive.
        """A worker still loading its weights would drop whatever it is given."""
        pods = [make_parsed_pod(pod_in_cell_index=0), make_parsed_pod(pod_in_cell_index=1, ready=False)]

        assert build_cell_info(pods).alive is False


class TestCellDeletion:
    # [CLASS DOC] Tests of how pods being deleted affect the cell info.
    def test_a_cell_holding_a_pod_under_deletion_is_not_alive(self):
        # [FUNCTION DOC] Two ready pods, one marked deleting: not alive.
        """A gracefully terminating pod stays Ready, so readiness alone reports a cell that is going away."""
        pods = [make_parsed_pod(pod_in_cell_index=0), make_parsed_pod(pod_in_cell_index=1, deleting=True)]

        assert build_cell_info(pods).alive is False

    def test_a_pod_entering_deletion_changes_the_cell_members_hash(self):
        # [FUNCTION DOC] The workers hash of the same two pods differs between the not-deleting and
        # the one-deleting variants.
        """Consumers reconcile on this hash, so a membership change it does not cover is never noticed."""
        before = [make_parsed_pod(pod_in_cell_index=0), make_parsed_pod(pod_in_cell_index=1)]
        after = [make_parsed_pod(pod_in_cell_index=0), make_parsed_pod(pod_in_cell_index=1, deleting=True)]

        assert build_cell_info(before).workers_hash != build_cell_info(after).workers_hash


class TestCellMeta:
    # [CLASS DOC] Tests of the cell metadata merged from pod metadata.
    def test_merges_the_facts_every_pod_of_the_cell_agrees_on(self):
        # [FUNCTION DOC] Both pods carry model_id glm; the cell meta is exactly that.
        """A cell reports one value per key, and its pods carry the same annotations."""
        pods = [make_parsed_pod(pod_in_cell_index=index, meta={"model_id": "glm"}) for index in range(2)]

        assert build_cell_info(pods).meta == {"model_id": "glm"}

    def test_refuses_a_cell_whose_pods_disagree_about_a_key(self):
        # [FUNCTION DOC] Pods with model_id glm and qwen make the computation raise an assertion
        # error mentioning model_id.
        """Whichever pod won would be whatever order the store happened to hand them back in."""
        pods = [
            make_parsed_pod(pod_in_cell_index=0, meta={"model_id": "glm"}),
            make_parsed_pod(pod_in_cell_index=1, meta={"model_id": "qwen"}),
        ]

        with pytest.raises(AssertionError, match="model_id"):
            build_cell_info(pods)


class TestWorkerInfos:
    # [CLASS DOC] Tests of the projection of pods to worker infos.
    def test_refuses_a_worker_whose_spec_declares_no_ports(self):
        # [FUNCTION DOC] A run with no ports raises an assertion error matching "declares no
        # ports".
        """A worker without a declared port cannot publish a callable address."""
        run = make_run(ports={}, worker_class=None)
        pod = make_parsed_pod(pod_in_cell_index=0, cell_size=1)

        with pytest.raises(AssertionError, match="declares no ports"):
            compute_worker_infos(CELL_ID, pods=[pod], run=run)

    def test_refuses_a_pod_with_neither_ip_nor_headless_service(self):
        # [FUNCTION DOC] A pod with no ip and no subdomain raises an assertion error matching
        # "neither a pod ip nor a headless service".
        """A pod without an IP or service cannot be projected to a valid host."""
        pod = make_parsed_pod(pod_in_cell_index=0, cell_size=1, pod_ip=None, subdomain=None)

        with pytest.raises(AssertionError, match="neither a pod ip nor a headless service"):
            compute_worker_infos(CELL_ID, pods=[pod], run=make_run())

    def test_fans_a_pod_out_into_one_worker_per_worker_it_serves(self):
        # [FUNCTION DOC] Two pods of two workers each (pod gpus 0 and 1): four infos named for
        # worker indices 0 to 3 of the pool, with gpu ids [0], [1], [0], [1] (each worker gets one
        # of its pod's gpus).
        """A pod runs several workers, and each of them is a Miles worker of its own."""
        pods = [make_parsed_pod(pod_in_cell_index=index, gpu_ids=(0, 1)) for index in range(2)]

        infos = build_worker_infos(pods, workers_per_pod=2)

        assert [info.name for info in infos] == [
            compute_worker_name(pool_id="engine", worker_in_cell_index=index) for index in range(4)
        ]
        assert [info.gpu_ids for info in infos] == [[0], [1], [0], [1]]

    def test_offsets_the_rpc_port_of_each_worker_the_way_its_process_binds_it(self):
        # [FUNCTION DOC] One pod with two workers and base rpc port 8000: the workers' rpc ports
        # are 8000 and 8001.
        """The workers of a pod share its ip, so only the port tells them apart."""
        infos = build_worker_infos(
            [make_parsed_pod(pod_in_cell_index=0, cell_size=1, gpu_ids=(0, 1))], workers_per_pod=2
        )

        assert [info.self_addrs["rpc"].port for info in infos] == [8000, 8001]

    def test_a_command_worker_is_reported_without_a_class_to_call_it_by(self):
        # [FUNCTION DOC] With no worker class the single info has no worker class and still lists
        # the rpc address host of the pod ip.
        """An engine pod runs no rpc server, so the dashboard reads its addresses but cannot call it."""
        run = make_run(worker_class=None)

        (info,) = compute_worker_infos(CELL_ID, pods=[make_parsed_pod(pod_in_cell_index=0, cell_size=1)], run=run)

        assert info.worker_class is None
        assert info.self_addrs["rpc"].host == "10.0.0.1"

    def test_an_ipv6_pod_is_addressed_in_brackets(self):
        # [FUNCTION DOC] A pod ip fd00::5 yields the host "[fd00::5]".
        """An unbracketed v6 address makes every url built from it unparseable."""
        pod = make_parsed_pod(pod_in_cell_index=0, cell_size=1, pod_ip="fd00::5")

        (info,) = compute_worker_infos(CELL_ID, pods=[pod], run=make_run())

        assert info.self_addrs["rpc"].host == "[fd00::5]"

    def test_refuses_a_cell_that_is_missing_a_pod(self):
        # [FUNCTION DOC] Only pod index 1 present (index 0 missing): assertion error matching
        # "missing pods".
        """Numbering workers off a gapped pod list would name workers that belong to another pod."""
        with pytest.raises(AssertionError, match="missing pods"):
            build_worker_infos([make_parsed_pod(pod_in_cell_index=1)])

    def test_refuses_a_cell_still_missing_its_last_pod(self):
        # [FUNCTION DOC] Only pod 0 present with expected size 2: assertion error matching "missing
        # pods".
        """A prefix check alone passes a half-arrived group, and the tail is what is actually missing."""
        with pytest.raises(AssertionError, match="missing pods"):
            build_worker_infos([make_parsed_pod(pod_in_cell_index=0, cell_size=2)])
