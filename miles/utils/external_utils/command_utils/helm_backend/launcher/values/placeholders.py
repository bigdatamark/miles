# [MODULE DOC]
# Sentinel and placeholder handling for pool entries. A values entry describes a whole pool, so
# values that differ per pod (pod index, base gpu id) are built with unlikely integer sentinels and
# afterwards replaced, only when they form a whole argument, by kubernetes variable references that
# the pod resolves at runtime. Also holds the leader address placeholder and the fixed cell index
# used when rendering.
from __future__ import annotations

from typing import NamedTuple

from miles.utils.workers import env_vars as worker_env_vars
from miles.utils.workers.worker_spec import BaseWorkerSpec

RENDERED_CELL_INDEX = 0

LEADER_ADDRESS_PLACEHOLDER = "$(LWS_LEADER_ADDRESS)"

WORKER_INDEX_SENTINEL = 987654321
_WORKER_INDEX_PLACEHOLDER = "$(LWS_WORKER_INDEX)"

_BASE_GPU_ID_SENTINEL = 987654322
_BASE_GPU_ID_PLACEHOLDER = f"$({worker_env_vars.BASE_GPU_ID_ENV_VAR})"


class _Substitution(NamedTuple):
    # [CLASS DOC] HELPER: One sentinel to placeholder rule, plus a short phrase naming what the
    # sentinel stands for, used in error messages.
    sentinel: int
    placeholder: str
    built_out_of: str


_SUBSTITUTIONS = (
    _Substitution(sentinel=WORKER_INDEX_SENTINEL, placeholder=_WORKER_INDEX_PLACEHOLDER, built_out_of="pod index"),
    _Substitution(sentinel=_BASE_GPU_ID_SENTINEL, placeholder=_BASE_GPU_ID_PLACEHOLDER, built_out_of="base gpu id"),
)


def real_or_sentinel_gpu_ids(spec: BaseWorkerSpec, *, is_sub_node: bool) -> list[int]:
    # [FUNCTION DOC] Gpu ids for one pod of the spec. The count is the spec's gpus per pod, but at
    # least one (so cpu-only pods still get one id). For sub-node pods the ids are all the base gpu
    # id sentinel (the real offset is only known at runtime); otherwise they are zero up to count.
    gpus_per_pod = max(1, spec.scheduling.gpus_per_pod())
    if is_sub_node:
        return [_BASE_GPU_ID_SENTINEL] * gpus_per_pod
    return list(range(gpus_per_pod))


def sentinels_to_placeholders(argv: list[str], spec: BaseWorkerSpec) -> list[str]:
    # [FUNCTION DOC] Returns a new argument list where every argument equal to a sentinel (as text)
    # is replaced by the matching placeholder. Raises an assertion error if a sentinel appears
    # inside a longer argument.
    for substitution in _SUBSTITUTIONS:
        sentinel = str(substitution.sentinel)
        _assert_sentinel_is_whole_token(argv, sentinel=sentinel, spec=spec, built_out_of=substitution.built_out_of)
        argv = [substitution.placeholder if argument == sentinel else argument for argument in argv]
    return argv


def sentinel_to_placeholder(value: str, spec: BaseWorkerSpec) -> str:
    # [FUNCTION DOC] Single-value variant of the argument list substitution.
    return sentinels_to_placeholders([value], spec)[0]


def _assert_sentinel_is_whole_token(
    argv: list[str], *, sentinel: str, spec: BaseWorkerSpec, built_out_of: str
) -> None:
    # [FUNCTION DOC] HELPER: Asserts no argument merely contains the sentinel text without being
    # equal to it (such a value could not be substituted whole). Error names the spec and offenders.
    embedded = [argument for argument in argv if sentinel in argument and argument != sentinel]
    assert not embedded, (
        f"Spec '{spec.name}' builds {embedded} out of its {built_out_of}; the value is substituted a whole "
        f"argument at a time, so it has to reach the command unchanged"
    )
