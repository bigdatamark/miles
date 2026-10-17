# Adapted from https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/trainer/ray/utils.py#L1
# [MODULE DOC] HELPER: Small utilities for Ray GPU handling: the list of Ray environment variables
# that, when set to a non-empty value, stop Ray from rewriting the visible-devices variable of a
# worker (one per accelerator type), a predicate to check whether any is set, and a function giving
# the current CUDA device's physical UUID.
import os

import torch


# Refer to
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/nvidia_gpu.py#L95-L96
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/amd_gpu.py#L102-L103
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/npu.py#L94-L95
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/hpu.py#L116-L117
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/neuron.py#L108-L109
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/tpu.py#L171-L172
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/intel_gpu.py#L97-L98
NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
    "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
    "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
    "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
]


def ray_noset_visible_devices(env_vars=os.environ):
    # [FUNCTION DOC] HELPER: True if any of the "do not set visible devices" variables has a truthy
    # (non-empty) value in the given environment mapping (default: the live process environment,
    # bound at definition time as the same mapping object). Note the text "0" also counts as set.
    return any(env_vars.get(env_var) for env_var in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST)


def get_physical_gpu_id():
    # [FUNCTION DOC] HELPER: Text form of the UUID of the current torch CUDA device (identifies the
    # physical GPU independent of visible-device remapping). Requires an initialized CUDA context.
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return str(props.uuid)
