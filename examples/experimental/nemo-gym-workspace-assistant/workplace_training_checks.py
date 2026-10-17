# [MODULE DOC] Pre-train-step hook for the Workplace example that asserts multi-token-prediction
# (MTP) auxiliary heads are completely absent from the Megatron model and that training starts
# from a fresh state initialized from the original HF checkpoint. Intended to be registered as the
# before-train-step hook; failures are assertion errors that abort the run.
"""Fail before a Workplace policy update if an auxiliary MTP head is present."""

import json
import logging
from argparse import Namespace
from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def before_train_step(
    args: Namespace,
    rollout_id: int,
    step_id: int,
    model: Sequence[torch.nn.Module],
    optimizer: Any,
    opt_param_scheduler: Any,
) -> None:
    # [FUNCTION DOC] Hook called before each training step with the parsed args, rollout and step
    # ids, the list of model chunks, and the optimizer and scheduler (the last two unused). For each
    # chunk it unwraps nested wrapper modules (following the inner module attribute repeatedly) to
    # the base model and asserts: the config's MTP layer count is unset; MTP forward processing is
    # not active; there is no MTP submodule; no parameter name contains "mtp" (case-insensitive).
    # Then asserts the args do not enable MTP training. On the very first step (rollout 0, step 0)
    # it additionally asserts: start rollout id is unset or zero; finetune is on and optimizer and
    # RNG state loading are disabled; the load path equals the HF checkpoint path. It then logs a
    # marker line with the distributed rank, no MTP layers, zero MTP parameters and a fresh-HF flag.
    # Returns nothing; raises assertion errors on any violation. Requires torch distributed to be
    # initialized when the logging branch runs.
    """Check the constructed model, including heads omitted from CLI config."""
    for chunk in model:
        module = chunk
        while hasattr(module, "module"):
            module = module.module
        config = module.config
        assert config.mtp_num_layers is None, f"MTP layers present: {config.mtp_num_layers}"
        assert not getattr(module, "mtp_process", False), "MTP forward is active"
        assert getattr(module, "mtp", None) is None, "MTP module is present"
        assert not any("mtp" in name.lower() for name, _ in module.named_parameters()), "MTP weights are present"
    assert not args.enable_mtp_training, "MTP training must be disabled"
    if rollout_id == 0 and step_id == 0:
        assert args.start_rollout_id in (None, 0), "Expected a fresh rollout cursor"
        assert args.finetune and args.no_load_optim and args.no_load_rng, "Expected fresh training state"
        assert args.load == args.hf_checkpoint, "Expected initialization from original HF weights"
        logger.info(
            "WORKPLACE_MTP_DISABLED %s",
            json.dumps({"rank": dist.get_rank(), "mtp_layers": None, "mtp_parameters": 0, "fresh_hf": True}),
        )
