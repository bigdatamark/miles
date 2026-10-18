# [MODULE DOC] GPU acceptance test that runs the client-side cookbook supervised and RL recipes
# (the multi-LoRA example client script in "both" mode, 2 steps) against a real Tinker-compatible
# gateway started locally. Registers a CUDA test (8-GPU H100 stage, estimated 400 seconds, labels
# lora, weight-update and multi-lora, hopper only). Run as a script: prepare the model, then
# execute.

"""Run the official cookbook SFT and RL recipes against a real Tinker gateway."""

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.lora.tinker_gateway import BASE_MODEL, prepare_gateway, running_gateway

from miles.utils.external_utils import command_utils

register_cuda_ci(
    est_time=400,
    suite="stage-c-8-gpu-h100",
    labels=["lora", "weight-update", "multi-lora"],
    hardware=["hopper"],
)


def execute():
    # [FUNCTION DOC] Starts the gateway (waiting until healthy), runs the client recipes script as
    # a subprocess with the gateway url, the base model and "both" mode for 2 steps, and stops the
    # gateway afterwards. Raises if the gateway fails to start or the client command fails.
    U = command_utils.default_config().create_backend()
    with running_gateway() as base_url:
        U.exec_command_cpu(
            "python examples/multi_lora/run_client_recipes.py "
            f"--base-url {base_url} --base-model {BASE_MODEL} --mode both --steps 2"
        )


if __name__ == "__main__":
    prepare_gateway()
    execute()
