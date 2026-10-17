# [MODULE DOC]
# Thin functional facade over the process-global routing replay manager. Defines stage name
# constants (fallthrough, record, replay forward, replay backward) and functions to enable replay
# from args, fill replay buffers from rollout data, choose the stage for log-prob computation,
# temporarily switch the stage, rewind forward cursors and reset all state.
from contextlib import contextmanager

from miles.backends.training_utils.replay.data import fill_replay_data, register_replay_list_sequential
from miles.utils.replay_base import routing_replay_manager

FALLTHROUGH = "fallthrough"
RECORD = "record"
REPLAY_FORWARD = "replay_forward"
REPLAY_BACKWARD = "replay_backward"


def uses_rollout_replay(args) -> bool:
    # [FUNCTION DOC] Whether routing decisions recorded by the rollout engine are replayed (the
    # use-rollout-routing-replay flag), as opposed to trainer-recorded replay.
    return args.use_rollout_routing_replay


def enable(args) -> bool:
    # [FUNCTION DOC] Configure the global manager: enabled = the use-routing-replay flag; the
    # replay-result check is enabled when replay is enabled and CI test mode is on; the registration
    # function is set to the sequential registration. Mutates the manager; returns whether replay
    # is enabled.
    routing_replay_manager.enabled = args.use_routing_replay
    routing_replay_manager.enable_check_replay_result = routing_replay_manager.enabled and args.ci_test
    routing_replay_manager.register_replay_list_func = register_replay_list_sequential
    return routing_replay_manager.enabled


def fill(args, models, data_iterators, num_microbatches, rollout_data, align=None) -> None:
    # [FUNCTION DOC] If rollout replay is not used, return. Otherwise fill the manager's replay
    # objects from the rollout data under the manager's data key (using the manager's registration
    # function, SP-region flag and indices-are-token-positions flag); this removes the data key from
    # the rollout data. Then, if an align callable is given, replaces every recorded index tensor
    # in every replay object with align(entry, -1), where -1 is the padding value.
    if not uses_rollout_replay(args):
        return

    fill_replay_data(
        args=args,
        models=models,
        data_iterator=data_iterators,
        num_microbatches=num_microbatches,
        rollout_data=rollout_data,
        data_key=routing_replay_manager.data_key,
        replay_list=routing_replay_manager.replays,
        register_replay_list_func=routing_replay_manager.register_replay_list_func,
        if_sp_region=routing_replay_manager.if_sp_region,
        indices_are_token_positions=routing_replay_manager.replay_indices_are_token_positions,
    )

    if align is None:
        return
    for replay in routing_replay_manager.replays:
        for i, entry in enumerate(replay.top_indices_list):
            replay.top_indices_list[i] = align(entry, -1)


def log_prob_stage(args) -> str:
    # [FUNCTION DOC] Stage to use for log-prob forward passes: fallthrough when the manager is
    # disabled; replay forward when rollout replay is used; otherwise record.
    if not routing_replay_manager.enabled:
        return FALLTHROUGH
    return REPLAY_FORWARD if uses_rollout_replay(args) else RECORD


@contextmanager
def stage(name: str):
    # [FUNCTION DOC] Context manager: sets the manager's stage to the given name for the duration of
    # the block and restores the previous stage on exit, even on exceptions.
    previous = routing_replay_manager.stage
    routing_replay_manager.stage = name
    try:
        yield
    finally:
        routing_replay_manager.stage = previous


def rewind() -> None:
    # [FUNCTION DOC] Clear the forward replay cursors/state in all replay objects of the manager so
    # replay can restart from the beginning.
    routing_replay_manager.clear_all_forward()


def reset() -> None:
    # [FUNCTION DOC] Clear all recorded data and state in all replay objects of the manager.
    routing_replay_manager.clear_all()
