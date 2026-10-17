# [MODULE DOC] Weights and Biases setup helpers: offline-mode detection, the primary process run
# creation (optionally logging in, random group suffix, shared mode), secondary processes attaching to
# the primary run in shared mode (optionally forwarding inference-engine metrics), config assembly, and
# the metric-to-step-key definitions common to both. Mutates environment variables and the args
# namespace (stores the run id) and global wandb state.
import logging
import os
from copy import deepcopy

import wandb
from wandb.sdk.lib.runid import generate_id

from miles.utils.env_report.launcher_report import read_launcher_report

logger = logging.getLogger(__name__)


def _is_offline_mode(args) -> bool:
    # [FUNCTION DOC] HELPER: If the wandb mode argument is non-empty, returns whether it equals
    # "offline"; otherwise returns whether the WANDB_MODE environment variable equals "offline".
    """Detect whether W&B should run in offline mode.

    Priority order:
    1) args.wandb_mode if provided
    2) WANDB_MODE environment variable
    """
    if args.wandb_mode:
        return args.wandb_mode == "offline"
    return os.environ.get("WANDB_MODE") == "offline"


def _wandb_settings(**kwargs):
    # [FUNCTION DOC] HELPER: Builds wandb settings with a 300 second init timeout plus the given
    # keyword settings.
    return wandb.Settings(init_timeout=300.0, **kwargs)


def init_wandb_primary(args):
    # [FUNCTION DOC] If use-wandb is falsy, sets the stored run id to none and returns. Otherwise: a
    # non-empty wandb mode argument overrides the WANDB_MODE environment variable (logging a message
    # for offline, disabled or online); when not offline and a key is given, logs in with the key and
    # host; group and run name are the group argument, or with the random-suffix option the group plus
    # underscore plus a generated short id and run name group-RANK_<rank>; builds init keywords (team,
    # project, group, name, config from the args plus whitelisted environment and the launcher
    # environment report, and the resume id with resume allowed when an id exists); settings are
    # offline mode, or shared mode as primary; creates the wandb directory if given; calls wandb init,
    # defines the common step metrics, and stores the new run id on the args.
    # KNOWN ISSUE: the wandb backend adapter forwards extra keyword arguments to this function, which
    # accepts none, so any extra keyword would raise a type error.
    if not args.use_wandb:
        args.wandb_run_id = None
        return

    # Set W&B mode if specified (overrides WANDB_MODE env var)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
        if args.wandb_mode == "offline":
            logger.info("W&B offline mode enabled. Data will be saved locally.")
        elif args.wandb_mode == "disabled":
            logger.info("W&B disabled mode enabled. No data will be logged.")
        elif args.wandb_mode == "online":
            logger.info("W&B online mode enabled. Data will be uploaded to cloud.")

    offline = _is_offline_mode(args)

    # Only perform explicit login when NOT offline
    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    # Prepare wandb init parameters
    # add random 6 length string with characters
    if args.wandb_random_suffix:
        group = args.wandb_group + "_" + generate_id()
        run_name = f"{group}-RANK_{args.rank}"
    else:
        group = args.wandb_group
        run_name = args.wandb_group

    # Prepare wandb init parameters
    init_kwargs = {
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "group": group,
        "name": run_name,
        "config": _compute_config_for_logging(args),
    }
    if args.wandb_run_id is not None:
        init_kwargs |= {"id": args.wandb_run_id, "resume": "allow"}

    # Configure settings based on offline/online mode
    if offline:
        init_kwargs["settings"] = _wandb_settings(mode="offline")
    else:
        init_kwargs["settings"] = _wandb_settings(mode="shared", x_primary=True)

    # Add custom directory if specified
    if args.wandb_dir:
        # Ensure directory exists to avoid backend crashes
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir
        logger.info(f"W&B logs will be stored in: {args.wandb_dir}")

    wandb.init(**init_kwargs)

    _init_wandb_common()

    # Set wandb_run_id in args for easy access throughout the training process
    args.wandb_run_id = wandb.run.id


def _compute_config_for_logging(args):
    # [FUNCTION DOC] HELPER: Deep-copies the args attribute mapping, adds an environment section with
    # only the SLURM job id variable if set, and, if the env-report argument is non-empty and the
    # launcher report can be read from it, adds that report under a launcher env report key. Returns
    # the mapping.
    output = deepcopy(args.__dict__)

    whitelist_env_vars = [
        "SLURM_JOB_ID",
        # We may insert more default values here, and may also allow users to configure a whitelist
    ]
    output["env_vars"] = {k: v for k, v in os.environ.items() if k in whitelist_env_vars}

    if env_report_raw := args.env_report:
        if launcher_report := read_launcher_report(env_report_raw):
            output["launcher_env_report"] = launcher_report

    return output


# https://docs.wandb.ai/guides/track/log/distributed-training/#track-all-processes-to-a-single-run
def init_wandb_secondary(args, router_addr=None):
    # [FUNCTION DOC] Returns immediately if the args have no run id. Otherwise applies the wandb mode
    # environment override, logs in when online with a key, and sets settings: offline, or shared mode
    # as non-primary that does not update the finish state; when inference-engine metrics are enabled
    # and a router address is given, adds the engine metrics endpoint (router address plus the
    # engine-metrics path) and an all-pass filter for stats collection. Calls wandb init with the same
    # run id, team, project, the raw args mapping as config, resume allowed and reinit true (and the
    # directory if configured), then defines the common step metrics.
    wandb_run_id = getattr(args, "wandb_run_id", None)
    if wandb_run_id is None:
        return

    # Set W&B mode if specified (same as primary)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    offline = _is_offline_mode(args)

    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    # Configure settings based on offline/online mode
    if offline:
        settings_kwargs = dict(mode="offline")
    else:
        settings_kwargs = dict(
            mode="shared",
            x_primary=False,
            x_update_finish_state=False,
        )

    if args.sglang_enable_metrics and router_addr is not None:
        logger.info(f"Forward SGLang metrics at {router_addr} to WandB.")
        settings_kwargs |= dict(
            x_stats_open_metrics_endpoints={
                "sgl_engine": f"{router_addr}/engine_metrics",
            },
            x_stats_open_metrics_filters={
                "sgl_engine.*": {},
            },
        )

    init_kwargs = {
        "id": wandb_run_id,
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "config": args.__dict__,
        "resume": "allow",
        "reinit": True,
        "settings": _wandb_settings(**settings_kwargs),
    }

    # Add custom directory if specified
    if args.wandb_dir:
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir

    wandb.init(**init_kwargs)

    _init_wandb_common()


def _init_wandb_common():
    # [FUNCTION DOC] HELPER: Declares step axes: train metrics against the train step; rollout,
    # multi-turn, pass-rate and perf metrics against the rollout step; eval metrics against the eval
    # step.
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("rollout/step")
    wandb.define_metric("rollout/*", step_metric="rollout/step")
    wandb.define_metric("multi_turn/*", step_metric="rollout/step")
    wandb.define_metric("passrate/*", step_metric="rollout/step")
    wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric="eval/step")
    wandb.define_metric("perf/*", step_metric="rollout/step")
