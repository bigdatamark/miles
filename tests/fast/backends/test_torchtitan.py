# [MODULE DOC] CPU tests for the torchtitan training backend utilities: argument validation (torch
# version floor, sequence length vs rotary tables, context parallelism for qwen3_5, unsupported
# flags), trainer config construction (tied embeddings under pipeline parallelism, lr schedule from
# miles flags, partial dtensor spmd backend, resume/save paths), the parallel state creation with
# stubbed distributed calls, the RL loss adapter's gradient scaling, a fingerprint guard on the
# upstream router forward that the routing-replay copy mirrors, and the GLM4.7 flash model spec vs
# its HF config. Tests needing the torchtitan package skip when it is not installed. Registered as a
# CPU test in the first CPU stage.
import json
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci

from miles.backends.torchtitan_utils import parallel as tp
from miles.backends.torchtitan_utils.arguments import validate_torchtitan_args

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])


def _args(**overrides) -> Namespace:
    # [FUNCTION DOC] HELPER: Builds a valid base args namespace for torchtitan validation (qwen3
    # model, sequence length 8192, all parallel sizes 1, max context 8192, max response 4096, no
    # ref update interval or debug dump, bf16, constant lr decay with no warmup fields); keyword
    # overrides replace entries. Returns the namespace.
    base = dict(
        titan_model_name="qwen3",
        seq_length=8192,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        rollout_max_context_len=8192,
        rollout_max_response_len=4096,
        ref_update_interval=None,
        save_debug_train_data=None,
        fp16=False,
        lr_decay_style="constant",
        lr_warmup_fraction=None,
        lr_wsd_decay_iters=None,
        lr_decay_iters=None,
    )
    return Namespace(**{**base, **overrides})


def test_the_backend_needs_torch_213(monkeypatch):
    # [FUNCTION DOC] Test: with the reported torch version 2.12.0, validation raises a value error
    # naming the torch 2.13 floor; with 2.13.0 it passes.
    monkeypatch.setattr(torch, "__version__", "2.12.0")
    with pytest.raises(ValueError, match="torch>=2.13"):
        validate_torchtitan_args(_args())
    monkeypatch.setattr(torch, "__version__", "2.13.0")
    validate_torchtitan_args(_args())


def test_the_sequence_must_cover_the_rotary_tables(monkeypatch):
    # [FUNCTION DOC] Test: with torch 2.13: a max rollout context (16384) above seq length (8192)
    # raises a value error mentioning seq-length; no context limit with a response limit equal to
    # the sequence length raises "no room for a prompt"; sequence 16384 with response 8192 passes.
    monkeypatch.setattr(torch, "__version__", "2.13.0")
    with pytest.raises(ValueError, match="seq-length"):
        validate_torchtitan_args(_args(seq_length=8192, rollout_max_context_len=16384))
    with pytest.raises(ValueError, match="no room for a prompt"):
        validate_torchtitan_args(_args(seq_length=8192, rollout_max_context_len=None, rollout_max_response_len=8192))
    validate_torchtitan_args(_args(seq_length=16384, rollout_max_context_len=None, rollout_max_response_len=8192))


def test_context_parallelism_is_rejected_for_qwen3_5(monkeypatch):
    # [FUNCTION DOC] Test: model qwen3_5 with context parallel size 2 raises a value error
    # mentioning context parallelism.
    monkeypatch.setattr(torch, "__version__", "2.13.0")
    with pytest.raises(ValueError, match="context parallelism"):
        validate_torchtitan_args(_args(titan_model_name="qwen3_5", context_parallel_size=2))


def test_unsupported_flags_are_rejected_rather_than_ignored(monkeypatch):
    # [FUNCTION DOC] Test: setting a ref update interval, or a debug train data dump path, each
    # raises a value error naming that flag.
    monkeypatch.setattr(torch, "__version__", "2.13.0")
    with pytest.raises(ValueError, match="ref-update-interval"):
        validate_torchtitan_args(_args(ref_update_interval=4))
    with pytest.raises(ValueError, match="save-debug-train-data"):
        validate_torchtitan_args(_args(save_debug_train_data="/tmp/dump"))


def _config_args(**overrides) -> Namespace:
    # [FUNCTION DOC] HELPER: Builds an args namespace for trainer config construction (adam, qwen3
    # 0.6B flavor, sequence 4096, unit parallel sizes, global batch 8, micro batch 1, grad clip 1,
    # lr 1e-6, min lr 0, no warmup, constant decay, adam betas 0.9/0.98, eps 1e-8, weight decay 0.1,
    # seed 1, no checkpointing, no save/load); keyword overrides replace entries.
    base = dict(
        optimizer="adam",
        titan_model_name="qwen3",
        titan_model_flavor="0.6B",
        seq_length=4096,
        dp_replicate_size=1,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        global_batch_size=8,
        micro_batch_size=1,
        clip_grad=1.0,
        lr=1e-6,
        min_lr=0.0,
        lr_warmup_iters=0,
        lr_decay_style="constant",
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_eps=1e-8,
        weight_decay=0.1,
        seed=1,
        gradient_checkpointing=False,
        save=None,
        load=None,
    )
    return Namespace(**{**base, **overrides})


def _checkpoint_dir(tmp_path, **config) -> str:
    # [FUNCTION DOC] HELPER: Creates the directory (with parents) if needed and writes a
    # config file inside it holding model type qwen3 plus the given extra entries. Returns the
    # directory path as a string.
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3", **config}))
    return str(tmp_path)


@pytest.fixture
def single_gpu_dims(monkeypatch):
    # [FUNCTION DOC] HELPER: Fixture; skips when torchtitan is missing, then patches the config
    # module's parallel-dims builder to return a single-GPU namespace (replicate 1, shard 1,
    # pipeline disabled) so configs build without a distributed setup.
    pytest.importorskip("torchtitan")
    from miles.backends.torchtitan_utils import config as titan_config

    monkeypatch.setattr(
        titan_config,
        "parallel_dims_from_config",
        lambda parallelism: SimpleNamespace(dp_replicate=1, dp_shard=1, pp_enabled=False),
    )


def test_a_tied_checkpoint_is_refused_under_pipeline_parallelism(tmp_path):
    # [FUNCTION DOC] Test: building a trainer config from a checkpoint with tied word embeddings and
    # pipeline size 2 raises a value error mentioning pipeline.
    pytest.importorskip("torchtitan")
    from miles.backends.torchtitan_utils.config import build_trainer_config

    hf = _checkpoint_dir(tmp_path, tie_word_embeddings=True)
    with pytest.raises(ValueError, match="pipeline"):
        build_trainer_config(
            _config_args(pipeline_model_parallel_size=2), hf_assets_path=hf, lr_total_steps=1, dump_subdir="x"
        )


def test_the_lr_schedule_follows_miles_flags_not_torchtitan_defaults(tmp_path, single_gpu_dims):
    # [FUNCTION DOC] Test: constant decay with 3 warmup iters over 10 steps gives scheduler warmup
    # steps 3 and min lr factor 1.0; cosine decay with lr 1e-6 and min lr 1e-7 gives decay type
    # cosine and min lr factor approximately 0.1.
    pytest.importorskip("torchtitan")
    from miles.backends.torchtitan_utils.config import build_trainer_config

    hf = _checkpoint_dir(tmp_path, tie_word_embeddings=False)
    constant = build_trainer_config(
        _config_args(lr_warmup_iters=3), hf_assets_path=hf, lr_total_steps=10, dump_subdir="x"
    )
    assert (constant.lr_scheduler.warmup_steps, constant.lr_scheduler.min_lr_factor) == (3, 1.0)
    cosine = build_trainer_config(
        _config_args(lr_decay_style="cosine", lr=1e-6, min_lr=1e-7),
        hf_assets_path=hf,
        lr_total_steps=10,
        dump_subdir="x",
    )
    assert cosine.lr_scheduler.decay_type == "cosine"
    assert cosine.lr_scheduler.min_lr_factor == pytest.approx(0.1)


def test_the_trainer_keeps_dtensor_only_on_the_model_parallel_axes(tmp_path, single_gpu_dims):
    # [FUNCTION DOC] Test: the built trainer config's parallelism spmd backend is the partial
    # dtensor backend.
    """torchtitan v0.3.0 defaults to the spmd_types backend, whose FSDP wrapping wants every
    parameter distributed over the full mesh first and rejects a plain-DP model."""
    pytest.importorskip("torchtitan")
    from miles.backends.torchtitan_utils.config import build_trainer_config

    config = build_trainer_config(
        _config_args(),
        hf_assets_path=_checkpoint_dir(tmp_path, tie_word_embeddings=False),
        lr_total_steps=10,
        dump_subdir="x",
    )
    assert config.parallelism.spmd_backend == "partial_dtensor"


class _Mesh:
    # [CLASS DOC] HELPER: Fake device mesh whose process group is a descriptive string naming its
    # size.
    def __init__(self, size: int):
        # [FUNCTION DOC] HELPER: Stores a string group label that encodes the size.
        self._group = f"pg(size={size})"

    def get_group(self):
        # [FUNCTION DOC] HELPER: Returns the string group label.
        return self._group


class _ParallelDims:
    # [CLASS DOC] HELPER: Fake parallel-dims object holding a mapping from mesh name to fake mesh.
    def __init__(self, meshes: dict[str, int]):
        # [FUNCTION DOC] HELPER: Creates a fake mesh per name from a mapping of name to size.
        self._meshes = {name: _Mesh(size) for name, size in meshes.items()}

    def get_optional_mesh(self, name):
        # [FUNCTION DOC] HELPER: Returns the mesh for the name, or none if absent.
        return self._meshes.get(name)

    def get_mesh(self, name):
        # [FUNCTION DOC] HELPER: Returns the mesh for the name; a missing name raises a key error.
        return self._meshes[name]


@pytest.fixture
def dist_stub():
    # [FUNCTION DOC] HELPER: Fixture yielding a mutable mapping from group label to size. While
    # active, the parallel module's distributed calls are patched: rank is always 0; world size is
    # looked up by group in the mapping (falling back to the "__world__" entry, then 1); creating a
    # group returns "self_group" with no backend or a "gloo_sub<ranks>" label otherwise; group ranks
    # are range(size); all-gather-object fills the output with copies of the local object; the
    # gloo group getter returns the label "gloo".
    sizes: dict = {}

    def get_world_size(group=None):
        # [FUNCTION DOC] HELPER: Looks up size by group label, falling back to the world entry or 1.
        return sizes.get(group, sizes.get("__world__", 1))

    with (
        patch.object(tp.dist, "get_rank", lambda group=None: 0),
        patch.object(tp.dist, "get_world_size", get_world_size),
        patch.object(
            tp.dist,
            "new_group",
            lambda ranks, backend=None: "self_group" if backend is None else f"gloo_sub{tuple(ranks)}",
        ),
        patch.object(tp.dist, "get_process_group_ranks", lambda group: list(range(sizes.get(group, 1)))),
        patch.object(
            tp.dist, "all_gather_object", lambda out, obj, group=None: out.__setitem__(slice(None), [obj] * len(out))
        ),
        patch.object(tp, "get_gloo_group", lambda: "gloo"),
    ):
        yield sizes


def _state(dist_stub, meshes: dict[str, int], world: int | None = None, **kwargs):
    # [FUNCTION DOC] HELPER: Registers sizes in the stub for every mesh group label; world defaults
    # to the loss mesh size (or 1); sets world, gloo group, self group (size 1) and the gloo
    # sub-group of the data-parallel ranks (size of the batch mesh, default 1). Then creates and
    # returns the torchtitan parallel state from fake parallel dims plus extra keyword arguments.
    for size in meshes.values():
        dist_stub[f"pg(size={size})"] = size
    world = world if world is not None else meshes.get("loss", 1)
    dist_stub["__world__"] = world
    dist_stub["gloo"] = world
    dist_stub["self_group"] = 1
    dp = meshes.get("batch", 1)
    dist_stub[f"gloo_sub{tuple(range(dp))}"] = dp
    return tp.create_titan_parallel_state(_ParallelDims(meshes), **kwargs)


def test_context_parallelism_stays_inside_the_trainer_and_absent_axes_are_trivial(dist_stub):
    # [FUNCTION DOC] Test: with meshes batch 4, loss 8, cp 2, the state's intra-dp size is 4,
    # intra-dp-with-cp size 4 and cp size 1 (context parallelism stays inside the trainer); the tp,
    # pp, ep, etp and independent-dp axes all have size 1 and rank 0.
    state = _state(dist_stub, {"batch": 4, "loss": 8, "cp": 2})
    assert (state.intra_dp.size, state.intra_dp_cp.size, state.cp.size) == (4, 4, 1)
    for axis in (state.tp, state.pp, state.ep, state.etp, state.indep_dp):
        assert (axis.size, axis.rank) == (1, 0)


def test_the_loss_adapter_undoes_the_summed_dp_and_cp_gradients():
    # [FUNCTION DOC] Test: an RL loss adapter with gradient scale 1/8 is armed with one batch and a
    # loss function returning the sum of predictions plus a metrics dictionary noting whether the
    # batch was the armed one, in training mode. Calling it on predictions of two entries each
    # valued 4 with dummy targets gives loss 1.0 (sum 8 times 1/8), and collecting
    # returns the one metrics dictionary with seen true.
    pytest.importorskip("torchtitan")
    from miles.backends.torchtitan_utils.loss import RLLossAdapter

    adapter = RLLossAdapter(RLLossAdapter.Config())
    adapter.set_gradient_scale(1.0 / 8)
    batch = {"tokens": torch.zeros(1)}
    adapter.arm([batch], lambda pred, b: (pred.sum(), {"seen": b is batch}), is_training=True)
    loss, _ = adapter(torch.full((2,), 4.0), torch.zeros(2, dtype=torch.long))
    assert loss.item() == pytest.approx(1.0)
    assert adapter.collect() == [{"seen": True}]


def test_resume_reads_from_load_and_writes_to_save(tmp_path, single_gpu_dims):
    # [FUNCTION DOC] Test: with a load root containing checkpoint dirs for steps 3 and 12 under the
    # backend/actor/checkpoint layout and a separate save root, the config dumps to the save root's
    # actor folder, resumes from the highest step (12), and loads a full (not model-only, not HF)
    # checkpoint with last-save not model-only. With an empty load root, there is no initial load
    # path and loading is model-only from HF format.
    pytest.importorskip("torchtitan")
    from miles.backends.torchtitan_utils.config import build_trainer_config

    hf = _checkpoint_dir(tmp_path / "hf", tie_word_embeddings=False)
    load_root = tmp_path / "load"
    for step in (3, 12):
        (load_root / "torchtitan" / "actor" / "checkpoint" / f"step-{step}").mkdir(parents=True)
    save_root = tmp_path / "save"

    config = build_trainer_config(
        _config_args(load=str(load_root), save=str(save_root)),
        hf_assets_path=hf,
        lr_total_steps=1,
        dump_subdir="actor",
    )
    assert config.dump_folder == str(save_root / "torchtitan" / "actor")
    assert config.checkpoint.initial_load_path == str(load_root / "torchtitan" / "actor" / "checkpoint" / "step-12")
    assert (config.checkpoint.initial_load_model_only, config.checkpoint.initial_load_in_hf) == (False, False)
    assert config.checkpoint.last_save_model_only is False

    fresh = build_trainer_config(
        _config_args(load=str(tmp_path / "empty"), save=str(save_root)),
        hf_assets_path=hf,
        lr_total_steps=1,
        dump_subdir="actor",
    )
    assert fresh.checkpoint.initial_load_path is None
    assert (fresh.checkpoint.initial_load_model_only, fresh.checkpoint.initial_load_in_hf) == (True, True)


def test_the_replayed_router_forward_mirrors_the_pinned_torchtitan_router():
    # [FUNCTION DOC] Test: hashes (sha256) the syntax-tree dump of the dedented source of the
    # upstream token-choice top-k router forward and asserts it equals a pinned fingerprint, so a
    # torchtitan upgrade that changes the forward forces the routing-replay copy to be re-synced.
    """routing_replay._token_router_forward is a copy of TokenChoiceTopKRouter.forward with the
    top-k swapped for the replayed one; a torchtitan bump that changes the upstream forward must
    re-sync the copy before this fingerprint is updated."""
    import ast
    import hashlib
    import inspect
    import textwrap

    moe = pytest.importorskip("torchtitan.models.common.moe")

    source = textwrap.dedent(inspect.getsource(moe.TokenChoiceTopKRouter.forward))
    fingerprint = hashlib.sha256(ast.dump(ast.parse(source)).encode()).hexdigest()
    assert fingerprint == "e3c6e10d721ebf5bdc8d968524cc204fbc3c74e775157b30248d6ec2f1aecef9"


_GLM47_FLASH_HF_CONFIG = dict(
    hidden_size=2048,
    num_hidden_layers=47,
    num_attention_heads=20,
    q_lora_rank=768,
    kv_lora_rank=512,
    qk_nope_head_dim=192,
    qk_rope_head_dim=64,
    v_head_dim=256,
    intermediate_size=10240,
    moe_intermediate_size=1536,
    n_routed_experts=64,
    n_shared_experts=1,
    num_experts_per_tok=4,
    first_k_dense_replace=1,
    routed_scaling_factor=1.8,
    norm_topk_prob=True,
    rope_theta=1000000,
    vocab_size=154880,
)

_GLM47_FLASH_CHECKPOINT_KEYS = [
    "lm_head.weight",
    "model.embed_tokens.weight",
    "model.norm.weight",
    "model.layers.{}.input_layernorm.weight",
    "model.layers.{}.post_attention_layernorm.weight",
    "model.layers.{}.self_attn.q_a_proj.weight",
    "model.layers.{}.self_attn.q_a_layernorm.weight",
    "model.layers.{}.self_attn.q_b_proj.weight",
    "model.layers.{}.self_attn.kv_a_proj_with_mqa.weight",
    "model.layers.{}.self_attn.kv_a_layernorm.weight",
    "model.layers.{}.self_attn.kv_b_proj.weight",
    "model.layers.{}.self_attn.o_proj.weight",
    "model.layers.{}.mlp.gate_proj.weight",
    "model.layers.{}.mlp.up_proj.weight",
    "model.layers.{}.mlp.down_proj.weight",
    "model.layers.{}.mlp.gate.weight",
    "model.layers.{}.mlp.gate.e_score_correction_bias",
    "model.layers.{}.mlp.experts.{}.gate_proj.weight",
    "model.layers.{}.mlp.experts.{}.up_proj.weight",
    "model.layers.{}.mlp.experts.{}.down_proj.weight",
    "model.layers.{}.mlp.shared_experts.gate_proj.weight",
    "model.layers.{}.mlp.shared_experts.up_proj.weight",
    "model.layers.{}.mlp.shared_experts.down_proj.weight",
]


def test_unknown_model_names_name_both_package_roots():
    # [FUNCTION DOC] Test: resolving a model spec for an unknown model name raises a value error
    # whose message names both the miles-provided and the torchtitan package import paths.
    pytest.importorskip("torchtitan")
    from miles.backends.torchtitan_utils.config import resolve_model_spec

    with pytest.raises(ValueError, match="miles.backends.torchtitan_utils.models.nope.*torchtitan.models.nope"):
        resolve_model_spec(_args(titan_model_name="nope", titan_model_flavor="x"))


def test_glm4_7_flash_flavor_matches_its_hf_config_and_its_checkpoint_keys():
    # [FUNCTION DOC] Test: resolves the glm4 moe lite 30B-A3B spec and checks against the pinned
    # HF config constants: hidden size, vocab size, layer count; attention head count, lora ranks,
    # nope/rope/value head dims, rope theta and no rope scaling; only the first dense-replace layers
    # lack a moe block; dense feed-forward width; routed expert count and per-expert hidden size;
    # shared expert width (moe width times shared count); router top-k, sigmoid scoring, route scale
    # and normalization flag; no mtp layers. Finally the state dict adapter's from-HF map must
    # contain every pinned checkpoint key template.
    pytest.importorskip("torchtitan")
    from miles.backends.torchtitan_utils.config import resolve_model_spec

    spec = resolve_model_spec(_args(titan_model_name="glm4_moe_lite", titan_model_flavor="30B-A3B"))
    hf = _GLM47_FLASH_HF_CONFIG
    model = spec.model
    assert (model.dim, model.vocab_size, len(model.layers)) == (
        hf["hidden_size"],
        hf["vocab_size"],
        hf["num_hidden_layers"],
    )
    attention = model.layers[0].attention
    assert attention.n_heads == hf["num_attention_heads"]
    assert (
        attention.q_lora_rank,
        attention.kv_lora_rank,
        attention.qk_nope_head_dim,
        attention.qk_rope_head_dim,
        attention.v_head_dim,
    ) == (hf["q_lora_rank"], hf["kv_lora_rank"], hf["qk_nope_head_dim"], hf["qk_rope_head_dim"], hf["v_head_dim"])
    assert attention.rope.theta == hf["rope_theta"] and attention.rope.scaling == "none"
    assert [layer.moe is None for layer in model.layers] == [i < hf["first_k_dense_replace"] for i in range(47)]
    assert model.layers[0].feed_forward.w1.out_features == hf["intermediate_size"]
    moe = model.layers[1].moe
    assert moe.num_experts == hf["n_routed_experts"]
    assert moe.routed_experts.inner_experts.hidden_dim == hf["moe_intermediate_size"]
    assert moe.shared_experts.w1.out_features == hf["moe_intermediate_size"] * hf["n_shared_experts"]
    assert (moe.router.top_k, moe.router.score_func, moe.router.route_scale, moe.router.route_norm) == (
        hf["num_experts_per_tok"],
        "sigmoid",
        hf["routed_scaling_factor"],
        hf["norm_topk_prob"],
    )
    assert model.mtp_layers == []

    adapter = spec.state_dict_adapter(model, None)
    assert set(_GLM47_FLASH_CHECKPOINT_KEYS) <= set(adapter.from_hf_map)
