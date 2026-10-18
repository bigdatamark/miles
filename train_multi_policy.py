# [MODULE DOC] Entry script for training several policies (one trainer per model id) that share
# one rollout executor and inference controller. One policy is the leader, whose configured
# rollout count defines the run length; followers run until the leader finishes. Save and eval
# points are synchronized across policies with parking barriers (followers park, the leader
# acts while all are parked). Multi-policy checkpoint state records the leader id and per-model
# rollout ids. Running it parses arguments and runs the coroutine under a disposer.
import asyncio
import copy
import itertools
import logging
import os
from argparse import Namespace
from pathlib import Path

from miles.backends.megatron_utils.megatron_config import resolve_megatron_config
from miles.ray.placement_group import create_rollout_components, maybe_start_api_server, update_weights
from miles.ray.specs.train import compute_trainer_configs
from miles.utils.arguments import parse_args
from miles.utils.async_utils import Disposer, wait_cancelling_pending_on_first_completion, with_disposer
from miles.utils.data import remove_rollout_data_refs
from miles.utils.ft_utils.mini_ft_controller import maybe_start_mini_ft_controller
from miles.utils.misc import should_run_periodic_action
from miles.utils.multi_policy.checkpoint_state import MultiPolicyCheckpointState
from miles.utils.multi_policy.parker import Parker
from miles.utils.multi_policy.utils import (
    TrainerInfo,
    assert_consistent_restore,
    create_trainers,
    define_policy_metric_groups,
    validate_multi_policy_args,
)
from miles.utils.orchestration_utils import init_orchestration_script
from miles.utils.workers.worker_handle import BaseWorkerHandle

logger = logging.getLogger(__name__)


async def train_multi_policy(args, *, disposer: Disposer) -> None:
    # [FUNCTION DOC] Top-level driver. Resolves the Megatron configuration, validates the
    # multi-policy arguments, initializes orchestration (handle unused), defines per-policy metric
    # groups, creates rollout components and one trainer per model (all registered for disposal),
    # asserts the restored state is consistent with the leader, starts the API server (mapping
    # each trainer id to its model's handle) and the optional mini fault-tolerance controller.
    # For each model in order: syncs initial weights (with start rollout id set to that trainer's
    # own) and optionally compares inference weights for that model. If eval is enabled, the
    # leader starts at round zero and eval is not skipped, runs an initial eval. Then creates two
    # barriers (save, eval) for followers (count = trainers minus one), a run-ended event, a shared
    # mapping of current rollout ids, and one concurrent task per policy; waits until the first
    # task completes, cancels the others, and sets the run-ended event on first completion.
    megatron_config = resolve_megatron_config(args)
    validate_multi_policy_args(args, megatron_config=megatron_config)
    _worker_manager = init_orchestration_script(args, disposer=disposer)

    define_policy_metric_groups(megatron_config)

    inference_controller, rollout_executor, num_rollout_per_epoch = await create_rollout_components(args)
    disposer.add(inference_controller, rollout_executor)

    trainers = await create_trainers(args, rollout_executor=rollout_executor)
    for trainer in trainers.values():
        disposer.add(trainer.handle)
    assert_consistent_restore(args, trainers=trainers, leader_model_id=megatron_config.leader_model_id)

    maybe_start_api_server(
        args,
        trainer_models={
            trainer_config.trainer_id: trainers[trainer_config.model_id].handle
            for trainer_config in compute_trainer_configs(args)
        },
        inference_controller=inference_controller,
    )
    maybe_start_mini_ft_controller(args)

    for model_id, trainer in trainers.items():
        await update_weights(
            _startup_args(args, trainer=trainer),
            trainer.handle,
            rollout_executor,
            inference_controller,
            trainer_model_id=model_id,
        )
        if args.check_weight_update_equal:
            await inference_controller.check_weights(
                action="compare",
                allow_quant_error=args.check_weight_update_allow_quant_error,
                selector=args.check_weight_update_selector,
                skip_list=args.check_weight_update_skip_list,
                model_id=model_id,
            )

    leader_start_rollout_id = trainers[megatron_config.leader_model_id].start_rollout_id
    if args.eval_interval is not None and leader_start_rollout_id == 0 and not args.skip_eval_before_train:
        await inference_controller.prepare_eval()
        await rollout_executor.eval(0)

    save_parker = Parker(num_followers=len(trainers) - 1)
    eval_parker = Parker(num_followers=len(trainers) - 1)
    run_ended = asyncio.Event()
    rollout_ids: dict[str, int] = {}
    tasks = [
        asyncio.create_task(
            _run_policy(
                args,
                trainer=trainer,
                is_leader=trainer.model_id == megatron_config.leader_model_id,
                trainers=trainers,
                inference_controller=inference_controller,
                rollout_executor=rollout_executor,
                save_parker=save_parker,
                eval_parker=eval_parker,
                run_ended=run_ended,
                rollout_ids=rollout_ids,
                num_rollout_per_epoch=num_rollout_per_epoch,
            )
        )
        for trainer in trainers.values()
    ]
    await wait_cancelling_pending_on_first_completion(tasks, on_first_completion=run_ended.set)


def _startup_args(args, *, trainer: TrainerInfo) -> Namespace:
    # [FUNCTION DOC] HELPER: Returns a shallow copy of the arguments whose start rollout id is
    # replaced by the trainer's own (restored) start id; the input is not modified.
    ans = copy.copy(args)
    ans.start_rollout_id = trainer.start_rollout_id
    return ans


async def _run_policy(
    args,
    *,
    trainer: TrainerInfo,
    is_leader: bool,
    run_ended: asyncio.Event,
    trainers: dict[str, TrainerInfo],
    inference_controller: BaseWorkerHandle,
    rollout_executor: BaseWorkerHandle,
    save_parker: Parker,
    eval_parker: Parker,
    rollout_ids: dict[str, int],
    num_rollout_per_epoch: int | None,
) -> None:
    # [FUNCTION DOC] HELPER: Loop for one policy. The leader iterates from its start id up to the
    # configured rollout count; followers iterate without bound from their start id and stop when
    # the run-ended event is set (checked at the top of each round). Per round: record the round
    # id in the shared mapping, prepare inference and fetch data for this model, train. The leader
    # then performs a possible global save (all parked); followers park at the save barrier. Every
    # update-interval rounds the policy syncs its weights to inference. The leader runs eval when
    # due while all followers are parked at the eval barrier; followers park there. The leader
    # stops early at the debug rollout count (measured from its own start id).
    model_id = trainer.model_id

    rollout_ids_iter = (
        range(trainer.start_rollout_id, args.num_rollout) if is_leader else itertools.count(trainer.start_rollout_id)
    )
    for rollout_id in rollout_ids_iter:
        if run_ended.is_set():
            return
        rollout_ids[model_id] = rollout_id
        await inference_controller.prepare_rollout(rollout_id, model_id=model_id)
        rollout_data_pack = await rollout_executor.get(rollout_id, trainer_model_id=model_id)
        await trainer.handle.train(rollout_id, rollout_data_pack)
        remove_rollout_data_refs(args, rollout_data_pack)

        if is_leader:
            await _maybe_save_globally(
                args,
                model_id=model_id,
                trainers=trainers,
                rollout_executor=rollout_executor,
                parker=save_parker,
                rollout_ids=rollout_ids,
                rollout_id=rollout_id,
                num_rollout_per_epoch=num_rollout_per_epoch,
            )
        else:
            await save_parker.maybe_park_follower()

        if (rollout_id + 1) % args.update_weights_interval == 0:
            await update_weights(
                args,
                trainer.handle,
                rollout_executor,
                inference_controller,
                rollout_id=rollout_id,
                trainer_model_id=model_id,
            )

        if is_leader:
            if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch, args.num_rollout):
                async with eval_parker.with_all_parked():
                    await inference_controller.prepare_eval()
                    await rollout_executor.eval(rollout_id)
        else:
            await eval_parker.maybe_park_follower()

        if (
            is_leader
            and (x := args.debug_exit_after_rollout) is not None
            and (rollout_id - trainer.start_rollout_id + 1) >= x
        ):
            logger.info(f"debug_exit_after_rollout={x} reached at rollout_id={rollout_id}, exiting")
            break


async def _maybe_save_globally(
    args,
    *,
    model_id: str,
    trainers: dict[str, TrainerInfo],
    rollout_executor: BaseWorkerHandle,
    parker: Parker,
    rollout_ids: dict[str, int],
    rollout_id: int,
    num_rollout_per_epoch: int | None,
) -> None:
    # [FUNCTION DOC] HELPER: Leader-only save step. Does nothing unless an external trigger file
    # exists or the periodic rule says a save is due. Then, with all followers parked, saves every
    # trainer's model synchronously under that trainer's own current round id, saves rollout
    # state for the leader's round id, and (if a save path is configured) writes the multi-policy
    # state (leader model id, copy of all current round ids) there. Afterwards removes the
    # trigger file if it was the reason.
    external_save = args.save_trigger_sentinel is not None and os.path.exists(args.save_trigger_sentinel)
    if not external_save and not should_run_periodic_action(
        rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
    ):
        return

    async with parker.with_all_parked():
        await asyncio.gather(
            *(
                trainer.handle.save_model(rollout_ids[trainer_model_id], force_sync=True)
                for trainer_model_id, trainer in trainers.items()
            )
        )
        await rollout_executor.save(rollout_id)
        if args.save is not None:
            MultiPolicyCheckpointState(leader_model_id=model_id, rollout_ids=dict(rollout_ids)).save(Path(args.save))

    if external_save:
        os.remove(args.save_trigger_sentinel)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(with_disposer(train_multi_policy, args))
