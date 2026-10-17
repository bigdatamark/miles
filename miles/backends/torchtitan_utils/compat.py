# [MODULE DOC] Runtime compatibility patches that let the torchtitan backend run on torch and
# torchtitan versions that differ from what it expects. Each patch checks whether it is needed and
# is a no-op otherwise, logs an info message when applied, and modifies third-party classes or
# modules globally (monkey patching), so they must be installed once before the trainer is built.
import inspect
import logging

logger = logging.getLogger(__name__)


def install() -> None:
    # [FUNCTION DOC] Applies all patches in order: the FSDP2 gradient-accumulation attribute-error
    # fix, the pipeline schedule microbatch-list step API, and the distributed set-timeout alias.
    _patch_fsdp2_grad_accumulation_attr_error()
    _patch_pipeline_schedule_microbatch_api()
    _shim_dist_set_timeout()


def _shim_dist_set_timeout() -> None:
    # [FUNCTION DOC] HELPER: If torch's distributed module lacks a set-timeout function, binds that
    # name to the internal process-group timeout setter and logs it; otherwise does nothing.
    import torch.distributed as dist

    if hasattr(dist, "set_timeout"):
        return
    from torch.distributed.distributed_c10d import _set_pg_timeout

    dist.set_timeout = _set_pg_timeout
    logger.info("Aliased torch.distributed.set_timeout to _set_pg_timeout")


def _patch_pipeline_schedule_microbatch_api() -> None:
    # [FUNCTION DOC] HELPER: Adds the newer (nightly) microbatch-list calling convention to the
    # single-stage and multi-stage pipeline schedules' step methods. Skipped if the single-stage
    # step already accepts the argument-microbatches parameter. Otherwise wraps both classes' step
    # methods (see the wrapper) and logs. Imports the torchtitan schedule classes lazily.
    import torch

    from torchtitan.distributed.pipeline_parallel import PipelineScheduleMulti, PipelineScheduleSingle

    if "arg_mbs" in inspect.signature(PipelineScheduleSingle.step).parameters:
        return

    def _make_step(original_step):
        # [FUNCTION DOC] HELPER: Returns a replacement step method wrapping the given original.
        def step(
            self,
            *args,
            arg_mbs=None,
            kwarg_mbs=None,
            target_mbs=None,
            target=None,
            losses=None,
            return_outputs=True,
            loss_kwargs=None,
            **kwargs
        ):
            # [FUNCTION DOC] HELPER: Replacement schedule step. If none of the argument, keyword
            # and target microbatch lists are given, delegates to the original step with the
            # positional arguments, target, losses, return-outputs flag (default true), loss keyword
            # arguments and any remaining keyword arguments. Otherwise: if the schedule has a
            # backward pass, requires autograd (unless the schedule says it does not) and gradients
            # are disabled, raises a runtime error telling the user to evaluate under no-grad; then
            # for every stage (the schedule's stage list, or its single stage) sets the stage's
            # backward flag from the schedule and clears its runtime states; finally runs the
            # schedule's internal microbatch stepping with the three microbatch lists, losses,
            # return-outputs flag and loss keyword arguments, returning its result.
            if arg_mbs is None and kwarg_mbs is None and target_mbs is None:
                return original_step(
                    self,
                    *args,
                    target=target,
                    losses=losses,
                    return_outputs=return_outputs,
                    loss_kwargs=loss_kwargs,
                    **kwargs,
                )
            if (
                self._has_backward
                and getattr(self, "_backward_requires_autograd", True)
                and not torch.is_grad_enabled()
            ):
                raise RuntimeError(
                    "step() requires gradients to be enabled for backward computation; "
                    "call eval() under torch.no_grad() instead."
                )
            stages = getattr(self, "_stages", None) or [self._stage]
            for stage in stages:
                stage.has_backward = self._has_backward
                stage.clear_runtime_states()
            return self._step_microbatches(
                arg_mbs, kwarg_mbs, target_mbs, losses, return_outputs, loss_kwargs=loss_kwargs
            )

        return step

    PipelineScheduleSingle.step = _make_step(PipelineScheduleSingle.step)
    PipelineScheduleMulti.step = _make_step(PipelineScheduleMulti.step)
    logger.info("Patched pipeline schedules with nightly's microbatch-list step API")


def _patch_fsdp2_grad_accumulation_attr_error() -> None:
    # [FUNCTION DOC] HELPER: Replaces FSDP2's per-parameter method that moves an unsharded gradient
    # into an accumulated gradient of the reduce dtype with a version that tolerates a missing
    # unsharded-parameter attribute. Skipped if the installed method's source text already contains
    # the safe attribute lookup. Logs when applied. Imports the FSDP internals lazily.
    from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam

    if "getattr" in inspect.getsource(FSDPParam.to_accumulated_grad_if_needed):
        return

    def to_accumulated_grad_if_needed(self) -> None:
        # [FUNCTION DOC] HELPER: Replacement method. Reads the unsharded parameter with a default of
        # none. Does nothing if the reduce dtype is unset, the unsharded parameter is missing or has
        # no gradient, or the gradient already has the reduce dtype. Otherwise detaches the gradient
        # from the parameter (sets it to none) and stores a copy cast to the reduce dtype as the
        # unsharded accumulated gradient. Mutates the parameter wrapper.
        unsharded_param = getattr(self, "_unsharded_param", None)
        if (
            self.reduce_dtype is None
            or unsharded_param is None
            or unsharded_param.grad is None
            or unsharded_param.grad.dtype == self.reduce_dtype
        ):
            return
        unsharded_grad = unsharded_param.grad
        unsharded_param.grad = None
        self.unsharded_accumulated_grad = unsharded_grad.to(self.reduce_dtype)

    FSDPParam.to_accumulated_grad_if_needed = to_accumulated_grad_if_needed
    logger.info("Patched FSDPParam.to_accumulated_grad_if_needed with the upstream getattr guard")
