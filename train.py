# [MODULE DOC] Entry script for synchronous (colocated or disaggregated) RL training. Running it
# parses the command-line arguments and runs the training coroutine under a disposer that
# cleans up the created components on exit. The coroutine builds the rollout components
# (inference controller, rollout executor) first because they determine the number of rollouts
# per epoch, then the actor (and optional critic) training models, optionally starts the API
# server and mini fault-tolerance controller, syncs initial weights to inference, optionally
# checks that the transferred weights are equal, and loops over rollouts: generate, optionally
# offload rollout memory, train (critic first when used), save periodically, hand weights back to
# inference, evaluate periodically, and optionally exit after a debug number of rollouts.
import asyncio
import logging
import os

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from miles.ray.placement_group import (
    create_rollout_components,
    create_training_models,
    maybe_start_api_server,
    update_weights,
)
from miles.ray.rollout.eval_dispatch import EvalDispatcher
from miles.utils.arguments import parse_args
from miles.utils.async_utils import Disposer, with_disposer
from miles.utils.data import remove_rollout_data_refs, remove_train_output_refs
from miles.utils.ft_utils.mini_ft_controller import maybe_start_mini_ft_controller
from miles.utils.lora.utils import lora_rollout_enabled
from miles.utils.misc import should_run_periodic_action
from miles.utils.orchestration_utils import init_orchestration_script

logger = logging.getLogger(__name__)


async def train(args, *, disposer: Disposer):
    # [FUNCTION DOC] The synchronous training driver; runs until all rollouts are done (or the
    # debug exit). Inputs: parsed arguments and the disposer that receives every created
    # component for teardown. Raises an assertion error if fully-async mode is on (needs the async
    # driver), or if the GPU memory-peak mode is set without both offload flags or together with
    # a critic. Behavior in order: initialize orchestration (the worker manager handle is kept
    # unused); create rollout components and training models and register them for disposal;
    # start optional services; push initial weights; optionally compare inference weights against
    # training (with the quantization-error allowance, selector and skip list from args); when
    # rollout offload is on, bring the KV cache back; create the evaluation dispatcher (its drain
    # is registered for disposal); run an eval-only pass if zero rollouts and an eval interval
    # are set; run a pre-training eval when there are rollouts to do, an eval interval is set and
    # not skipped (round zero uses the HF checkpoint path; resumed runs evaluate the previous
    # round). Per rollout round: prepare rollout, fetch data, offload inference memory (GPU
    # memory-peak mode: KV cache, then onload training, then weights; otherwise CUDA graph memory
    # plus KV cache and/or weights depending on the offload level string), train critic (offload
    # after) then actor once past the critic-only steps and pass critic values externally, or
    # train the actor alone; drop data references; save when an external trigger file exists
    # (removed after saving) or the periodic rule fires; if more rollouts remain or an eval is due,
    # move training memory off GPU (and in the memory-peak mode clear memory, offload the
    # gradient buffer for LoRA rollouts and onload inference weights first), sync weights to
    # inference and restore KV cache; run eval if due (forced on the last round); stop early once
    # the debug rollout count since the start is reached.
    assert not args.fully_async, "--fully-async requires the async driver: run train_async.py"
    _worker_manager = init_orchestration_script(args, disposer=disposer)

    if args.colocate_memory_peak_device == "gpu":
        assert (
            args.offload_train and args.offload_rollout
        ), "--colocate-memory-peak-device gpu requires --offload-train and --offload-rollout"
        assert not args.use_critic, "--colocate-memory-peak-device gpu is not wired for the critic path"

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

    if args.offload_rollout:
        await inference_controller.onload_kv()

    eval_dispatcher = EvalDispatcher(args, actor_model, rollout_executor)
    disposer.add(eval_dispatcher.drain)

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        await inference_controller.prepare_eval()
        await eval_dispatcher.dispatch(0, hf_dir=args.hf_checkpoint)

    async def offload_train():
        # [FUNCTION DOC] HELPER: Does nothing with a critic (critic offload is handled in the
        # loop). Otherwise offloads the actor if train offload is on, else just clears its memory.
        if args.use_critic:
            return
        if args.offload_train:
            await actor_model.offload()
        else:
            await actor_model.clear_memory()

    async def save(rollout_id, force_sync=False):
        # [FUNCTION DOC] HELPER: Saves checkpoints for a round. Synchronous saving is forced on
        # the last round. Saves the actor unless a critic exists and the round is still in the
        # critic-only phase; saves the critic if present; then saves rollout-side state.
        force_sync = force_sync or rollout_id == args.num_rollout - 1

        async def save_training_model(model):
            # [FUNCTION DOC] HELPER: Onloads the model first if critic and train offload are
            # both on, saves it, then offloads it again under the same condition.
            if args.use_critic and args.offload_train:
                await model.onload()
            await model.save_model(rollout_id, force_sync=force_sync)
            if args.use_critic and args.offload_train:
                await model.offload()

        if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps):
            await save_training_model(actor_model)
        if args.use_critic:
            await save_training_model(critic_model)
        await rollout_executor.save(rollout_id)

    if args.num_rollout > args.start_rollout_id and args.eval_interval is not None and not args.skip_eval_before_train:
        await inference_controller.prepare_eval()
        if args.start_rollout_id == 0:
            await eval_dispatcher.dispatch(0, hf_dir=args.hf_checkpoint)
        else:
            await eval_dispatcher.dispatch(args.start_rollout_id - 1)

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        await inference_controller.prepare_rollout(rollout_id)
        rollout_data_pack = await rollout_executor.get(rollout_id)

        if args.offload_rollout:
            if args.colocate_memory_peak_device == "gpu":
                await inference_controller.offload_kv()
                await actor_model.onload()
                await inference_controller.offload_weights()
            else:
                offload_tags = [GPU_MEMORY_TYPE_CUDA_GRAPH]
                if "kv_cache" in args.offload_rollout_level:
                    offload_tags.append(GPU_MEMORY_TYPE_KV_CACHE)
                if "weight" in args.offload_rollout_level:
                    offload_tags.append(GPU_MEMORY_TYPE_WEIGHTS)
                await inference_controller.offload(tags=offload_tags)

        if args.use_critic:
            values = await critic_model.train(rollout_id, rollout_data_pack)
            if args.offload_train:
                await critic_model.offload()
            if rollout_id >= args.num_critic_only_steps:
                await actor_model.train(rollout_id, rollout_data_pack, external_data=values)
                if args.offload_train:
                    await actor_model.offload()
            remove_train_output_refs(values)
        else:
            await actor_model.train(rollout_id, rollout_data_pack)
        remove_rollout_data_refs(args, rollout_data_pack)

        external_save = args.save_trigger_sentinel is not None and os.path.exists(args.save_trigger_sentinel)
        if external_save or should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        ):
            await save(rollout_id, force_sync=external_save)
            if external_save:
                os.remove(args.save_trigger_sentinel)

        # One predicate for both blocks: the handoff below exists to feed this eval on the last rollout.
        eval_due = should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch, args.num_rollout)
        if rollout_id + 1 < args.num_rollout or eval_due:
            if args.colocate_memory_peak_device == "gpu":
                await actor_model.clear_memory()
                if lora_rollout_enabled(args):
                    await actor_model.offload_grad_buffer()
                await inference_controller.onload_weights()
                await offload_train()
            else:
                await offload_train()
                if args.offload_rollout:
                    await inference_controller.onload_weights()
            await update_weights(args, actor_model, rollout_executor, inference_controller, rollout_id=rollout_id)
            if args.offload_rollout:
                await inference_controller.onload_kv()

        if eval_due:
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
