# [MODULE DOC] CPU tests that on-policy-distillation per-token fields (teacher log-probs or a
# precomputed reverse KL) are context-parallel sliced identically to rollout log-probs when the
# rollout data is loaded (packed layout cp 1 and 2, padded layout cp 2; expected kept response
# indices are fixed per case), stay fp32 on CPU, and feed the OPD advantage adjustment with the
# expected values; and that multimodal rollout expansion under cp 2 gathers the precomputed
# reverse KL tensor across context-parallel ranks once.
from argparse import Namespace

import pytest
import torch

from miles.backends.training_utils.data import context_parallel, multimodal
from miles.backends.training_utils.data import rollout as data_utils
from miles.backends.training_utils.loss.hub.opd import apply_opd_kl_to_advantages
from miles.backends.training_utils.parallel import GroupInfo, ParallelState

_ROLLOUT_LOG_PROBS = torch.tensor([-0.2, -1.3, -0.7, -2.1, -0.4, -3.2, -1.8])
_OPD_VALUES = torch.tensor([0.1, 0.9, -0.3, 1.7, -1.1, 0.4, 2.3])


def _parallel_state(*, cp_size: int, cp_rank: int = 0) -> ParallelState:
    # [FUNCTION DOC] HELPER: Builds a real parallel state where all groups are trivial (rank 0, size
    # 1) except context parallel and intra-dp-with-cp, which use the given rank (default 0) and
    # size; no process groups.
    trivial_group = GroupInfo(rank=0, size=1, group=None)
    return ParallelState(
        intra_dp=trivial_group,
        intra_dp_cp=GroupInfo(rank=cp_rank, size=cp_size, group=None),
        cp=GroupInfo(rank=cp_rank, size=cp_size, group=None),
        tp=trivial_group,
        pp=trivial_group,
        ep=trivial_group,
        etp=trivial_group,
        indep_dp=trivial_group,
    )


def _args(qkv_format: str) -> Namespace:
    # [FUNCTION DOC] HELPER: Returns args with the given layout, pad multiplier 16, no compress
    # ratios, and witness, true-on-policy, bf16 and fp16 all off.
    return Namespace(
        enable_witness=False,
        qkv_format=qkv_format,
        data_pad_size_multiplier=16,
        compress_ratios=[],
        true_on_policy_mode=False,
        bf16=False,
        fp16=False,
    )


def _load_rollout_data(
    monkeypatch: pytest.MonkeyPatch,
    *,
    qkv_format: str,
    cp_size: int,
    cp_rank: int,
    opd_key: str,
) -> dict:
    # [FUNCTION DOC] HELPER: Builds one sample (11 tokens, 7 response positions with ones mask,
    # rollout log-probs and the OPD field under the given key, both length 7 with fixed values),
    # patches the rollout processing function to return it, patches the parallel state getter in
    # both the rollout and context-parallel modules, and the cuda current-device query to cpu; then
    # runs the rollout data loader and returns the loaded rollout data dictionary.
    parallel_state = _parallel_state(cp_size=cp_size, cp_rank=cp_rank)
    rollout_data = {
        "tokens": [list(range(11))],
        "loss_masks": [[1] * 7],
        "total_lengths": [11],
        "response_lengths": [7],
        "rollout_log_probs": [_ROLLOUT_LOG_PROBS.tolist()],
        opd_key: [_OPD_VALUES.clone()],
    }

    monkeypatch.setattr(data_utils, "process_rollout_data", lambda *args, **kwargs: (rollout_data, object()))
    monkeypatch.setattr(data_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(context_parallel, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))

    loaded_rollout_data, _store_get_result = data_utils.get_rollout_data(_args(qkv_format), object())
    return loaded_rollout_data


@pytest.mark.parametrize("opd_key", ["teacher_log_probs", "opd_reverse_kl"])
@pytest.mark.parametrize(
    ("qkv_format", "cp_size", "cp_rank", "expected_indices"),
    [
        ("thd", 1, 0, [0, 1, 2, 3, 4, 5, 6]),
        ("thd", 2, 0, [6]),
        ("thd", 2, 1, [0, 1, 2, 3, 4, 5]),
        ("bshd", 2, 0, [0]),
        ("bshd", 2, 1, [1, 2, 3, 4, 5, 6]),
    ],
)
def test_sglang_opd_response_fields_follow_rollout_log_prob_cp_slice(
    monkeypatch: pytest.MonkeyPatch,
    opd_key: str,
    qkv_format: str,
    cp_size: int,
    cp_rank: int,
    expected_indices: list[int],
) -> None:
    # [FUNCTION DOC] Test (parametrized over OPD field name and five layout/cp/rank cases): after
    # loading, both rollout log-probs and the OPD field equal the fixed source vectors indexed by
    # the expected kept indices; the OPD field is fp32 on CPU. Then applies the OPD advantage
    # adjustment (sglang type, coefficient 0.5) with unit advantages and student log-probs equal to
    # the field plus 0.25: for teacher log-probs the advantages become 0.875 everywhere (1 minus
    # 0.5 times 0.25); for a precomputed reverse KL they become 1 minus 0.5 times the field.
    rollout_data = _load_rollout_data(
        monkeypatch,
        qkv_format=qkv_format,
        cp_size=cp_size,
        cp_rank=cp_rank,
        opd_key=opd_key,
    )

    expected_indices_tensor = torch.tensor(expected_indices)
    torch.testing.assert_close(
        rollout_data["rollout_log_probs"][0],
        _ROLLOUT_LOG_PROBS[expected_indices_tensor],
    )
    torch.testing.assert_close(
        rollout_data[opd_key][0],
        _OPD_VALUES[expected_indices_tensor],
    )
    assert rollout_data[opd_key][0].dtype == torch.float32
    assert rollout_data[opd_key][0].device.type == "cpu"

    advantages = [torch.ones(len(expected_indices), dtype=torch.float32)]
    student_log_probs = [rollout_data[opd_key][0] + 0.25]
    apply_opd_kl_to_advantages(
        Namespace(opd_type="sglang", opd_kl_coef=0.5),
        rollout_data,
        advantages,
        student_log_probs,
    )

    if opd_key == "teacher_log_probs":
        torch.testing.assert_close(advantages[0], torch.full_like(advantages[0], 0.875))
    else:
        torch.testing.assert_close(
            advantages[0],
            1.0 - 0.5 * rollout_data[opd_key][0],
        )


def test_multimodal_cp_reslices_precomputed_opd_reverse_kl(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] Test: one multimodal sample (media token plus two tokens, response length 2,
    # grid inputs, precomputed reverse KL of two values). With cp 2 patched in, the all-gather
    # function replaced by a recorder returning its input and the slicer replaced by identity,
    # the in-place multimodal expansion gathers exactly one tensor, which is the reverse KL tensor
    # object itself.
    rollout_data = {
        "tokens": [torch.tensor([multimodal.KIMI_VL_MEDIA_TOKEN_ID, 1, 2])],
        "loss_masks": [torch.ones(2, dtype=torch.int)],
        "total_lengths": [3],
        "response_lengths": [2],
        "multimodal_train_inputs": [{"grid_thws": torch.tensor([[1, 4, 4]])}],
        "opd_reverse_kl": [torch.tensor([0.1, 0.2])],
    }
    gathered_keys = []

    monkeypatch.setattr(multimodal, "get_parallel_state", lambda: _parallel_state(cp_size=2))

    def _all_gather(value: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Recorder replacing the cp all-gather; appends the value to a list
        # and returns it unchanged.
        gathered_keys.append(value)
        return value

    monkeypatch.setattr(multimodal, "all_gather_with_cp", _all_gather)
    monkeypatch.setattr(multimodal, "slice_log_prob_with_cp", lambda value, *args, **kwargs: value)

    multimodal.expand_multimodal_rollout_data_in_place(rollout_data)

    assert len(gathered_keys) == 1
    assert gathered_keys[0] is rollout_data["opd_reverse_kl"][0]
