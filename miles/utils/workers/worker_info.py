# [MODULE DOC]
# Wire model describing one live worker, exchanged between the managers, providers and the
# registration layer.
from __future__ import annotations

from pydantic import ConfigDict

from miles.utils.pydantic_utils import StrictBaseModel
from miles.utils.workers.worker_spec import NamedHostAndPorts


class WorkerInfo(StrictBaseModel):
    # [CLASS DOC]
    # Strict model (unknown fields forbidden, arbitrary types allowed). Fields: worker name; the
    # generation of its incarnation (increases when its cell is relaunched); its named endpoints; its
    # gpu ids; and the worker class path, present only when the worker is served over rpc.
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str
    generation: int
    self_addrs: NamedHostAndPorts
    gpu_ids: list[int]
    worker_class: str | None = None
