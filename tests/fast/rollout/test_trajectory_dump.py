# [MODULE DOC] Tests for the debug trajectory dump of the rollout executor. Checks that trajectory
# rows carry the raw conversation messages with sample and group index, reward and prompt (samples
# without messages are skipped; non-numeric rewards become null), and that saving debug rollout data
# writes a line-per-sample JSON sidecar file, strips messages from the tensor-serialized sample file,
# flattens evaluation datasets into an eval-prefixed sidecar, writes no sidecar when there are no
# conversations, and keeps messages in the main file when the sidecar path is unset. Uses real
# temporary directories; no GPU or network.
import argparse
import json

import pytest
import torch

from miles.ray.rollout.debug_data import save_debug_rollout_data, trajectory_rows
from miles.utils.types import Sample


def make_sample(**kwargs):
    # [FUNCTION DOC] HELPER: Builds a sample with defaults index 7, group index 2, prompt "rendered
    # prompt" and completed status; keyword arguments override or add fields.
    defaults = dict(index=7, group_index=2, prompt="rendered prompt", status=Sample.Status.COMPLETED)
    return Sample(**{**defaults, **kwargs})


MESSAGES = [
    {"role": "user", "content": "q"},
    {"role": "assistant", "content": "a"},
    {"role": "tool", "name": "get_year", "content": "2026"},
    {"role": "assistant", "content": "done"},
]


def test_rows_carry_the_raw_conversation():
    # [FUNCTION DOC] A sample with reward 1.0 and the sample conversation in its metadata yields
    # exactly one row with sample index 7, group index 2, reward 1.0, the rendered prompt, and the
    # messages list unchanged.
    sample = make_sample(reward=1.0, metadata={"messages": MESSAGES})
    (row,) = trajectory_rows([sample])
    assert row["sample_index"] == 7 and row["group_index"] == 2
    assert row["reward"] == 1.0 and row["prompt"] == "rendered prompt"
    assert row["messages"] == MESSAGES


def test_rows_skip_samples_without_a_conversation():
    # [FUNCTION DOC] Of two samples, only the one with messages in its metadata produces a row; the
    # other is skipped.
    with_messages = make_sample(index=1, metadata={"messages": MESSAGES})
    without = make_sample(index=2)
    rows = trajectory_rows([without, with_messages])
    assert [row["sample_index"] for row in rows] == [1]


def test_non_numeric_reward_becomes_null():
    # [FUNCTION DOC] A mapping-valued reward is written as null in the row.
    sample = make_sample(reward={"score": 1}, metadata={"messages": MESSAGES})
    (row,) = trajectory_rows([sample])
    assert row["reward"] is None


@pytest.fixture
def dump_args(tmp_path):
    # [FUNCTION DOC] HELPER: Fixture returning args whose rollout-data save path template is a .pt
    # file per rollout id under a rollout_data subdirectory of the temp path and whose trajectory
    # save path template is a .jsonl file per rollout id under a trajectory subdirectory.
    return argparse.Namespace(
        save_debug_rollout_data=str(tmp_path / "rollout_data" / "{rollout_id}.pt"),
        save_debug_trajectory_data=str(tmp_path / "trajectory" / "{rollout_id}.jsonl"),
    )


def test_save_writes_sidecar_and_strips_messages_from_pt(dump_args, tmp_path):
    # [FUNCTION DOC] Saving two samples with messages for rollout 5 (training, not eval) writes a
    # trajectory file with two lines whose sample indices are 0 and 1, and a loaded rollout-data file
    # whose samples have no messages entry in their metadata.
    samples = [make_sample(index=i, metadata={"messages": MESSAGES}) for i in range(2)]
    save_debug_rollout_data(dump_args, samples, 5, evaluation=False)
    lines = (tmp_path / "trajectory" / "5.jsonl").read_text().splitlines()
    assert [json.loads(line)["sample_index"] for line in lines] == [0, 1]
    saved = torch.load(tmp_path / "rollout_data" / "5.pt", weights_only=False)
    assert all("messages" not in s["metadata"] for s in saved["samples"])


def test_save_eval_flattens_datasets(dump_args, tmp_path):
    # [FUNCTION DOC] In eval mode the data is a mapping from dataset name to a mapping with a
    # samples list. Saving for rollout 1 writes a file named with an eval prefix holding one line
    # whose messages equal the conversation.
    data = {"math": {"samples": [make_sample(metadata={"messages": MESSAGES})]}}
    save_debug_rollout_data(dump_args, data, 1, evaluation=True)
    (line,) = (tmp_path / "trajectory" / "eval_1.jsonl").read_text().splitlines()
    assert json.loads(line)["messages"] == MESSAGES


def test_no_conversations_means_no_file(dump_args, tmp_path):
    # [FUNCTION DOC] Saving a sample without messages for rollout 9 still writes the main data file
    # but creates no trajectory directory or file.
    save_debug_rollout_data(dump_args, [make_sample()], 9, evaluation=False)
    assert (tmp_path / "rollout_data" / "9.pt").exists()
    assert not (tmp_path / "trajectory").exists()


def test_messages_stay_in_pt_when_sidecar_disabled(dump_args, tmp_path):
    # [FUNCTION DOC] With the trajectory save path unset, saving a sample with messages for rollout
    # 8 keeps the messages inside the main data file's sample metadata.
    dump_args.save_debug_trajectory_data = None
    save_debug_rollout_data(dump_args, [make_sample(metadata={"messages": MESSAGES})], 8, evaluation=False)
    saved = torch.load(tmp_path / "rollout_data" / "8.pt", weights_only=False)
    assert saved["samples"][0]["metadata"]["messages"] == MESSAGES
