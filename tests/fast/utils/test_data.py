# [MODULE DOC] Unit tests for the data utilities: the prompt Dataset (only structured, multimodal
# prompts go through the media processor) and the helpers that release object-store references held
# by train step outputs and rollout data packs. A recording in-memory store replaces the process-wide
# object store singleton for the reference tests. CPU only.
from __future__ import annotations

import json
from typing import Any

import pytest

from miles.backends.training_utils.types import TrainStepOutcome, TrainStepOutput
from miles.utils import object_store
from miles.utils.data import Dataset, RolloutDataPack, remove_rollout_data_refs, remove_train_output_refs
from miles.utils.object_store import (
    BaseObjectStore,
    ObjectStoreGetResult,
    StoreObjectRef,
    ValueSpec,
    _MooncakeStoreObjectRef,
)


class _RecordingProcessor:
    # [CLASS DOC] HELPER: Fake multimodal processor that records every prompt it is asked to extract
    # media from and always answers with one image and no videos.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Starts with an empty list of seen prompts.
        self.prompts = []

    def extract_media(self, prompt):
        # [FUNCTION DOC] HELPER: Records the prompt and returns a dict with images (a single
        # placeholder "image") and videos (empty list).
        self.prompts.append(prompt)
        return {"images": ["image"], "videos": []}


def test_mixed_dataset_uses_processor_only_for_structured_prompts(tmp_path) -> None:
    # [FUNCTION DOC] Writes a two-line JSON lines file: a plain text prompt and a structured chat
    # prompt (a user message with an image part and a text part). Builds a Dataset with the recording
    # processor, no tokenizer, no max length, prompt key "prompt" and no chat template. Asserts the
    # text sample keeps its prompt and has no multimodal inputs; the structured sample keeps its
    # prompt and gets the processor's media dict; and the processor saw only the structured prompt.
    text_prompt = "Fix the bug"
    multimodal_prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "image.png"},
                {"type": "text", "text": "Describe it"},
            ],
        }
    ]
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text(
        "\n".join(
            [
                json.dumps({"prompt": text_prompt}),
                json.dumps({"prompt": multimodal_prompt}),
            ]
        )
        + "\n"
    )
    processor = _RecordingProcessor()

    dataset = Dataset(
        str(prompt_path),
        tokenizer=None,
        processor=processor,
        max_length=None,
        prompt_key="prompt",
        apply_chat_template=False,
    )

    assert dataset.samples[0].prompt == text_prompt
    assert dataset.samples[0].multimodal_inputs is None
    assert dataset.samples[1].prompt == multimodal_prompt
    assert dataset.samples[1].multimodal_inputs == {
        "images": ["image"],
        "videos": [],
    }
    assert processor.prompts == [multimodal_prompt]


def _ref(payload: Any) -> StoreObjectRef:
    # [FUNCTION DOC] HELPER: Builds a Mooncake-style object reference wrapping the given payload.
    return _MooncakeStoreObjectRef(payload=payload)


class _RecordingStore(BaseObjectStore):
    # [CLASS DOC] HELPER: Fake object store that only records which references were removed.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Starts with an empty removed list.
        self.removed: list[StoreObjectRef] = []

    def put(self, value: Any, value_spec: dict[str, ValueSpec] | None = None) -> StoreObjectRef:
        # [FUNCTION DOC] HELPER: Returns a reference wrapping the value; stores nothing. The value
        # spec is ignored.
        return _ref(value)

    def get(self, ref: StoreObjectRef) -> ObjectStoreGetResult:
        # [FUNCTION DOC] HELPER: Not supported; always raises a not-implemented error.
        raise NotImplementedError

    def remove(self, ref: StoreObjectRef) -> None:
        # [FUNCTION DOC] HELPER: Appends the reference to the removed list.
        self.removed.append(ref)


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> _RecordingStore:
    # [FUNCTION DOC] HELPER: Fixture that creates a recording store and installs it as the module
    # singleton of the object store module (restored after the test); returns it.
    instance = _RecordingStore()
    monkeypatch.setattr(object_store, "_INSTANCE", instance)
    return instance


class TestRemoveTrainOutputRefs:
    # [CLASS DOC] Tests of remove_train_output_refs: releases the object-store reference held in the
    # values field of each train step output.
    def test_every_shipped_ref_is_released(self, store: _RecordingStore):
        # [FUNCTION DOC] Two outputs holding two references: the store's removed list equals those
        # references in order.
        """Nothing else frees these objects, so a missed ref leaks for the whole run under mooncake."""
        refs = [_ref("a"), _ref("b")]

        remove_train_output_refs([TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=ref) for ref in refs])

        assert store.removed == refs

    def test_a_worker_that_shipped_nothing_is_skipped(self, store: _RecordingStore):
        # [FUNCTION DOC] Outputs with values none, a reference, none: only the reference is removed.
        """Only pp-last-stage critic workers ship values, so the rest carry None and must not reach the store."""
        ref = _ref("a")

        remove_train_output_refs(
            [
                TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=None),
                TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=ref),
                TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=None),
            ]
        )

        assert store.removed == [ref]


class TestRemoveRolloutDataRefs:
    # [CLASS DOC] Tests of remove_rollout_data_refs: releases the reference (or list of references)
    # held by a rollout data pack; the first positional argument is passed as none here.
    def test_the_reference_the_rollout_shipped_is_released(self, store: _RecordingStore):
        # [FUNCTION DOC] A pack with a single data reference results in exactly that reference being
        # removed.
        """The driver is the only process that frees a rollout, once per step, or the store fills up."""
        ref = _ref("rollout")

        remove_rollout_data_refs(None, rollout_data_pack=RolloutDataPack(sample_indices=[0], data_ref=ref))

        assert store.removed == [ref]

    def test_every_shard_of_a_split_rollout_is_released(self, store: _RecordingStore):
        # [FUNCTION DOC] A pack whose data reference is a list of two references results in both being
        # removed, in order.
        """Without --delay-split-train-data-by-dp there is one object per dp rank, and each one pins memory."""
        refs = [_ref("a"), _ref("b")]

        remove_rollout_data_refs(None, rollout_data_pack=RolloutDataPack(sample_indices=[0], data_ref=refs))

        assert store.removed == refs

    def test_a_pack_that_carries_no_data_never_reaches_the_store(self, store: _RecordingStore):
        # [FUNCTION DOC] A default (empty) pack removes nothing: the removed list stays empty.
        """A pack without a data ref ships no object, and asking the store to free None would raise."""
        remove_rollout_data_refs(None, rollout_data_pack=RolloutDataPack())

        assert store.removed == []
