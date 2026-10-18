# [MODULE DOC] Entry script for asynchronous (non-colocated) RL training: generation of the next
# rollout overlaps training on the current one, and weights are pushed to inference every
# configured interval of rounds. Running it parses arguments and runs the training coroutine
# under a disposer for cleanup. See the training function for the algorithm.
import asyncio
import logging
import os

from miles.ray.placement_group import (
    create_rollout_components,
    create_training_models,
    maybe_start_api_server,
    update_weights,
)
from miles.ray.rollout.eval_dispatch import EvalDispatcher
from miles.utils.arguments import parse_args, validate_async_off_policy_correction
from miles.utils.async_utils import Disposer, eager_create_task, with_disposer
from miles.utils.data import remove_rollout_data_refs, remove_train_output_refs
from miles.utils.ft_utils.mini_ft_controller import maybe_start_mini_ft_controller
from miles.utils.misc import should_run_periodic_action
from miles.utils.orchestration_utils import init_orchestration_script

logger = logging.getLogger(__name__)


# The framework supports other asynchronous approaches such as fully async (see miles/rollout/fully_async_rollout.py).
async def train(args, *, disposer: Disposer):
    # [FUNCTION DOC] Asynchronous training driver. Asserts colocation is off and validates the
    # off-policy correction settings for async mode. Setup mirrors the synchronous driver (rollout
    # components first, then training models, optional services, initial weight sync, optional
    # weight-equality check, eval dispatcher, pre-training eval only for a fresh run with an eval
    # interval and not skipped). No offloading logic: training and inference use separate GPUs.
    # Loop per round: take the already-started generation for this round (a future created one
    # round earlier); compute whether a next round exists and whether a weight update is due
    # after this round (every update-interval rounds, counting from one); start generating the
    # next round immediately, unless fully-async mode is on and a weight update is due (the drain
    # is deferred until after the update so the new weights are used); train critic then actor
    # (past the critic-only steps, passing critic values) or the actor alone; drop data references;
    # save when an external trigger file exists or the periodic rule fires (forced synchronous on
    # trigger or last round; saves actor, critic if present, and rollout state; removes the
    # trigger file); on weight-update rounds, in non-fully-async mode first wait for the in-flight
    # generation to finish (so weights do not change mid-generation) and clear its future, then
    # sync weights, then start the deferred generation if one was deferred; run eval when due
    # (forced on the last round); stop early at the debug rollout count.
    # Note: on a non-fully-async weight-update round the awaited next-round data is stored in the
    # current-round variable and the future is cleared, so the next iteration uses that data
    # directly instead of awaiting a future.
    assert not args.colocate, "Colocation is not supported for async training."
    validate_async_off_policy_correction(args)
    _worker_manager = init_orchestration_script(args, disposer=disposer)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    inference_controller, rollout_executor, num_rollout_per_epoch = await create_rollout_components(args)
    disposer.add(inference_controller, rollout_executor)

    # create the actor and critic models
    actor_model, critic_model = await create_training_models(args, rollout_executor)
    disposer.add(critic_model, actor_model)

    maybe_start_api_server(args, trainer_models={"actor": actor_model}, inference_controller=inference_controller)
    maybe_start_mini_ft_controller(args)

    # always update weight first so that sglang has the loaded weights from training.
    await update_weights(args, actor_model, rollout_executor, inference_controller)

    if args.check_weight_update_equal:
        await inference_controller.check_weights(
            action="compare",
            allow_quant_error=args.check_weight_update_allow_quant_error,
            selector=args.check_weight_update_selector,
            skip_list=args.check_weight_update_skip_list,
        )

    eval_dispatcher = EvalDispatcher(args, actor_model, rollout_executor)
    disposer.add(eval_dispatcher.drain)

    if args.eval_interval is not None and args.start_rollout_id == 0 and not args.skip_eval_before_train:
        await inference_controller.prepare_eval()
        await eval_dispatcher.dispatch(0, hf_dir=args.hf_checkpoint)

    async def save_training_model(model, rollout_id, force_sync):
        # [FUNCTION DOC] HELPER: Onloads the model first if critic and train offload are on,
        # saves it for the round (optionally synchronously), then offloads it again.
        if args.use_critic and args.offload_train:
            await model.onload()
        await model.save_model(rollout_id, force_sync=force_sync)
        if args.use_critic and args.offload_train:
            await model.offload()

    async def prepare_and_generate(rollout_id):
        # [FUNCTION DOC] HELPER: Prepares the inference side for the round, then fetches and
        # returns that round's rollout data reference. Run as an eagerly started task.
        await inference_controller.prepare_rollout(rollout_id)
        return await rollout_executor.get(rollout_id)

    # async train loop.
    rollout_data_next_future = await eager_create_task(prepare_and_generate(args.start_rollout_id))
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # Sync the last generation
        if rollout_data_next_future is not None:
            rollout_data_curr_ref = await rollout_data_next_future

        has_next_rollout = rollout_id + 1 < args.num_rollout
        weight_update_due = (rollout_id + 1) % args.update_weights_interval == 0

        # A fully-async producer keeps generating without a pending get(). When
        # weights will change, defer the next drain so it uses the new version.
        defer_next_drain = args.fully_async and has_next_rollout and weight_update_due
        if has_next_rollout and not defer_next_drain:
            rollout_data_next_future = await eager_create_task(prepare_and_generate(rollout_id + 1))

        if args.use_critic:
            values = await critic_model.train(rollout_id, rollout_data_curr_ref)
            if args.offload_train:
                await critic_model.offload()
            if rollout_id >= args.num_critic_only_steps:
                await actor_model.train(rollout_id, rollout_data_curr_ref, external_data=values)
                if args.offload_train:
                    await actor_model.offload()
            remove_train_output_refs(values)
        else:
            await actor_model.train(rollout_id, rollout_data_curr_ref)
        remove_rollout_data_refs(args, rollout_data_curr_ref)

        external_save = args.save_trigger_sentinel is not None and os.path.exists(args.save_trigger_sentinel)
        if external_save or should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        ):
            force_sync = external_save or rollout_id == args.num_rollout - 1
            await save_training_model(actor_model, rollout_id, force_sync)
            if args.use_critic:
                await save_training_model(critic_model, rollout_id, force_sync)
            await rollout_executor.save(rollout_id)
            if external_save:
                os.remove(args.save_trigger_sentinel)

        if weight_update_due:
            if not args.fully_async:
                # sync generate before update weights to prevent update weight in the middle of generation
                rollout_data_curr_ref = (await x) if (x := rollout_data_next_future) is not None else None
                rollout_data_next_future = None
            await update_weights(args, actor_model, rollout_executor, inference_controller, rollout_id=rollout_id)
            if defer_next_drain:
                rollout_data_next_future = await eager_create_task(prepare_and_generate(rollout_id + 1))

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch, args.num_rollout):
            await inference_controller.prepare_eval()
            await eval_dispatcher.dispatch(rollout_id, force=rollout_id == args.num_rollout - 1)

        if (
            args.debug_exit_after_rollout is not None
            and (rollout_id - args.start_rollout_id + 1) >= args.debug_exit_after_rollout
        ):
            logger.info(
                "debug_exit_after_rollout=%d reached at rollout_id=%d, exiting",
                args.debug_exit_after_rollout,
                rollout_id,
            )
            break


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(with_disposer(train, args))
