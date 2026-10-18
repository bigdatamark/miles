# [MODULE DOC] GPU acceptance test of multi-tenant LoRA serving through a real Tinker-compatible
# gateway: runs the multi-tenant example client script in multi mode with 4 concurrent clients,
# rank-8 adapters, 12 steps, learning rate 1e-3 and 24 max sampled tokens, which has the tenants
# learn distinct markers and sample their own adapter snapshots. Registers a CUDA test (8-GPU
# H100 stage, estimated 400 seconds, multi-lora label, hopper only). Run as a script: prepare
# the model, then execute.

"""Concurrent tenants learn distinct markers and sample their own adapter snapshots."""

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.lora.tinker_gateway import BASE_MODEL, prepare_gateway, running_gateway

from miles.utils.external_utils import command_utils

register_cuda_ci(
    est_time=400,
    suite="stage-c-8-gpu-h100",
    labels=["multi-lora"],
    hardware=["hopper"],
)


def execute():
    # [FUNCTION DOC] Starts the gateway (waiting until healthy), runs the multi-tenant example
    # client as a subprocess with the gateway url and base model, and stops the gateway
    # afterwards. Raises if the gateway fails to start or the client command fails.
    U = command_utils.default_config().create_backend()
    with running_gateway() as base_url:
        U.exec_command_cpu(
            "python examples/multi_lora/run_multi_tenant_example.py "
            f"--base-url {base_url} --base-model {BASE_MODEL} "
            "--mode multi --clients 4 --lora-rank 8 --steps 12 --lr 1e-3 --max-tokens 24"
        )


if __name__ == "__main__":
    prepare_gateway()
    execute()
