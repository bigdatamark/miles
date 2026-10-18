# [MODULE DOC] Unit tests for multi-model sglang configuration parsing (the update-weights flag and
# per-model server groups) and for the helper that builds a model's generate URL from the router
# settings. Registers itself as a stage-a CPU CI test with a 60 second estimate. Imports of the
# code under test are done lazily inside helpers and tests.
"""Unit tests for SglangConfig multi-model parsing with update_weights."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import pytest
import yaml


def _write_yaml(data: dict, tmp_path) -> str:
    # [FUNCTION DOC] HELPER: Dumps the given mapping as YAML into a file named config.yaml inside
    # the temporary directory (writes to disk, overwriting) and returns the file path as a string.
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(data))
    return str(path)


def _resolve(path: str, *, rollout_num_gpus: int, hf_checkpoint: str = "/path/to/model"):
    # [FUNCTION DOC] HELPER: Builds a fake argument namespace (config path, no prefill servers, the
    # given rollout GPU count at one GPU per engine, 8 GPUs per node, no eval GPUs, engines
    # started, the given checkpoint path, and colocation, offload, debug and critic options all
    # off, with one actor node of 8 GPUs) and returns the result of resolving the sglang config.
    # The code under test is imported lazily here.
    from argparse import Namespace

    from miles.backends.sglang_utils.sglang_config import resolve_sglang_config

    args = Namespace(
        sglang_config=path,
        prefill_num_servers=None,
        rollout_num_gpus=rollout_num_gpus,
        rollout_num_gpus_per_engine=1,
        num_gpus_per_node=8,
        eval_num_gpus=0,
        starts_inference_engines=True,
        hf_checkpoint=hf_checkpoint,
        offload_rollout=False,
        debug_train_only=False,
        debug_rollout_only=False,
        colocate=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        critic_num_nodes=0,
        critic_num_gpus_per_node=0,
        use_critic=False,
        critic_train_only=False,
    )
    return resolve_sglang_config(args)


class TestSglangConfigUpdateWeights:
    # [CLASS DOC] Tests of resolving a YAML sglang config with several models.
    def test_update_weights_explicit_false(self, tmp_path):
        # [FUNCTION DOC] Writes a config with an actor model (update weights true, 4 regular GPUs)
        # and a ref model (update weights false, its own model path, 2 regular GPUs), resolves it
        # with 6 rollout GPUs, and asserts two models in order with the expected names, update
        # flags and the ref model path.
        """Models with update_weights: false should be parsed correctly."""
        path = _write_yaml(
            {
                "sglang": [
                    {
                        "name": "actor",
                        "update_weights": True,
                        "engine_groups": [{"worker_type": "regular", "num_gpus": 4}],
                    },
                    {
                        "name": "ref",
                        "update_weights": False,
                        "model_path": "/path/to/ref",
                        "engine_groups": [{"worker_type": "regular", "num_gpus": 2}],
                    },
                ]
            },
            tmp_path,
        )
        config = _resolve(path, rollout_num_gpus=6)
        assert len(config.models) == 2
        assert config.models[0].name == "actor"
        assert config.models[0].update_weights is True
        assert config.models[1].name == "ref"
        assert config.models[1].update_weights is False
        assert config.models[1].model_path == "/path/to/ref"

    def test_multi_model_total_gpus(self, tmp_path):
        # [FUNCTION DOC] Writes a config whose two models declare server groups of 8 and 4 GPUs,
        # resolves it with 12 rollout GPUs, and asserts the GPU counts summed over all groups of
        # all models equal 12.
        """Group gpu counts should sum across all models."""
        path = _write_yaml(
            {
                "sglang": [
                    {
                        "name": "actor",
                        "server_groups": [{"worker_type": "regular", "num_gpus": 8}],
                    },
                    {
                        "name": "ref",
                        "update_weights": False,
                        "server_groups": [{"worker_type": "regular", "num_gpus": 4}],
                    },
                ]
            },
            tmp_path,
        )
        config = _resolve(path, rollout_num_gpus=12)
        assert sum(g.num_gpus for m in config.models for g in m.server_groups) == 12


class TestGetModelUrl:
    # [CLASS DOC] Tests of the model URL helper, which maps a model name to the router address.
    def test_get_model_url_basic(self):
        # [FUNCTION DOC] With a default router and per-model routers for actor and ref, asserts
        # each model gets its own router host and port with the default generate path, and that a
        # custom path argument replaces the generate path.
        """get_model_url should return the correct URL for a named model."""
        from argparse import Namespace

        from miles.rollout.sglang_rollout import get_model_url

        args = Namespace(
            sglang_router_ip="10.0.0.1",
            sglang_router_port=3000,
            sglang_model_routers={
                "actor": ("10.0.0.1", 3000),
                "ref": ("10.0.0.1", 3001),
            },
        )
        assert get_model_url(args, "actor") == "http://10.0.0.1:3000/generate"
        assert get_model_url(args, "ref") == "http://10.0.0.1:3001/generate"
        assert get_model_url(args, "ref", "/v1/chat/completions") == "http://10.0.0.1:3001/v1/chat/completions"

    def test_get_model_url_fallback(self):
        # [FUNCTION DOC] Asks for a model name missing from the per-model routers and asserts the
        # default router address with the generate path is returned.
        """get_model_url should fall back to default router if model not found."""
        from argparse import Namespace

        from miles.rollout.sglang_rollout import get_model_url

        args = Namespace(
            sglang_router_ip="10.0.0.1",
            sglang_router_port=3000,
            sglang_model_routers={"actor": ("10.0.0.1", 3000)},
        )
        assert get_model_url(args, "unknown") == "http://10.0.0.1:3000/generate"

    def test_get_model_url_no_routers(self):
        # [FUNCTION DOC] With the per-model routers mapping unset (none), asserts the default
        # router address with the generate path is returned for any model name.
        """get_model_url should work when sglang_model_routers is left at its parser default."""
        from argparse import Namespace

        from miles.rollout.sglang_rollout import get_model_url

        args = Namespace(
            sglang_router_ip="10.0.0.1",
            sglang_router_port=3000,
            sglang_model_routers=None,
        )
        assert get_model_url(args, "anything") == "http://10.0.0.1:3000/generate"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
