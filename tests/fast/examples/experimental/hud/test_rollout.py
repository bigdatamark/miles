# [MODULE DOC]
# Offline tests (skipped unless the hud package is importable) for the HUD example's rollout
# helpers. The stitch function turns per-turn records (prompt token ids, output token ids,
# per-token log probabilities) into one training sequence: tokens, a loss mask (1 for policy-emitted
# tokens, 0 for re-rendered scaffolding), log probabilities (0.0 where not emitted), and the prompt
# length; it keeps the longest prefix of turns whose prompts extend the previous history, resyncs
# when a re-render only canonicalised whitespace (using a fake tokenizer with a tiny vocabulary),
# and returns nothing when no turn survives. The vision inputs function checks image token counts
# against screenshot counts using a fake image processor. The failed-sample function must shape a
# write-off sample so the trainer can still index it.
"""Offline tests: no GPU, no network, no Daytona, no live model.

rollout.py's translation: per-turn server token ids -> one training
sequence with masks (including whitespace resync), the vision inputs
verified against it, and the shape of a write-off.
The HUD side of the seam is faked at exactly the boundary the real code
touches.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

hud = pytest.importorskip("hud", reason="pip install hud -- the recipe's one extra dependency")

from examples.experimental.hud.rollout import _stitch, _vision_inputs


def _turn(prompt, output, logprobs=None):
    # [FUNCTION DOC] HELPER: Builds a fake turn record with the given prompt token ids, output
    # token ids, output log probabilities (default -0.1 per output token) and no prompt chunks.
    return SimpleNamespace(
        prompt_token_ids=prompt,
        output_token_ids=output,
        output_logprobs=logprobs if logprobs is not None else [-0.1] * len(output),
        prompt_chunks=None,
    )


def test_stitch_two_turns_masks_only_policy_tokens():
    # [FUNCTION DOC] Turn one prompt [1,2,3] output [10,11]; turn two prompt extends it with new
    # tokens 4,5 and outputs [12]. Asserts the stitched tokens, prompt length 3, mask [1,1,0,0,1],
    # log probs -0.1 for the first outputs and 0.0 for the added scaffolding tokens, and equal
    # lengths of mask and log probs to the response part.
    """The mask is the training signal's boundary: 1 on what the policy
    emitted, 0 on the scaffolding and screenshots re-rendered into turn 2."""
    t0 = _turn([1, 2, 3], [10, 11])
    t1 = _turn([1, 2, 3, 10, 11, 4, 5], [12])
    tokens, mask, logprobs, start = _stitch([t0, t1])
    assert tokens == [1, 2, 3, 10, 11, 4, 5, 12]
    assert start == 3
    assert mask == [1, 1, 0, 0, 1]  # out0, out0, delta, delta, out1
    assert logprobs[:2] == [-0.1, -0.1] and logprobs[2:4] == [0.0, 0.0]
    assert len(mask) == len(tokens) - start == len(logprobs)


class _WsTokenizer:
    # [CLASS DOC] HELPER: Fake tokenizer with a small vocabulary mapping a few ids to whitespace or
    # punctuation-plus-whitespace strings; other ids decode to an angle-bracketed number.
    """Fake tokenizer for resync tests: ids decode to fixed strings."""

    VOCAB = {271: "\n\n", 198: "\n", 220: " ", 50: ".\n\n", 51: ".\n"}

    def decode(self, ids, **kwargs):
        # [FUNCTION DOC] HELPER: Concatenates the vocabulary string of each id (extra keyword arguments ignored).
        return "".join(self.VOCAB.get(i, f"<{i}>") for i in ids)


def test_stitch_resyncs_when_the_rerender_canonicalizes_whitespace():
    # [FUNCTION DOC] The model emitted a double newline token that the next prompt re-renders as a
    # single newline. With the fake tokenizer, asserts the stitch adopts the re-render's token
    # (mask 0, log prob 0.0), keeps both turns, and prompt length 2.
    """The model emits a blank line before its tool call; the parser + chat
    template re-render it as a single newline. That is canonicalization, not a
    history rewrite, so the stitch adopts the re-render's whitespace (masked
    0) and keeps the rest of the episode."""
    t0 = _turn([1, 2], [10, 271, 11])  # output ends "...\n\n<tool_call>"
    t1 = _turn([1, 2, 10, 198, 11, 4, 5], [12])  # re-rendered with "\n"
    tokens, mask, logprobs, start = _stitch([t0, t1], tokenizer=_WsTokenizer())
    assert tokens == [1, 2, 10, 198, 11, 4, 5, 12]
    assert start == 2
    # out0, adopted ws (0), out0, delta, delta, out1
    assert mask == [1, 0, 1, 0, 0, 1]
    assert logprobs[1] == 0.0  # the adopted token carries no logprob


def test_stitch_resync_refuses_a_real_rewrite():
    # [FUNCTION DOC] The next prompt replaces output token 11 with a non-whitespace token 99;
    # asserts only turn one is kept (tokens [1,2,10,11], mask [1,1]).
    """Non-whitespace divergence must still end the stitch, tokenizer or not."""
    t0 = _turn([1, 2], [10, 11])
    t1 = _turn([1, 2, 10, 99, 4], [12])  # 11 -> 99 is a real rewrite
    tokens, mask, logprobs, start = _stitch([t0, t1], tokenizer=_WsTokenizer())
    assert tokens == [1, 2, 10, 11]  # turn 0 only
    assert mask == [1, 1]


def test_stitch_resync_refuses_when_the_remainder_shifts():
    # [FUNCTION DOC] Whitespace differs and the token after it also changes; asserts the stitch
    # keeps turn one exactly as emitted and drops the second turn.
    """Whitespace at the divergence is not enough: everything after it must
    realign exactly, otherwise the whole tail was rewritten."""
    t0 = _turn([1, 2], [10, 271, 11])
    t1 = _turn([1, 2, 10, 198, 77, 4], [12])  # after the ws, 11 -> 77
    tokens, mask, logprobs, start = _stitch([t0, t1], tokenizer=_WsTokenizer())
    assert tokens == [1, 2, 10, 271, 11]  # turn 0 kept as emitted


def test_stitch_resyncs_punctuation_merged_whitespace_tokens():
    # [FUNCTION DOC] Emitted token for period plus double newline versus re-rendered period plus
    # single newline (both non-pure whitespace tokens). Asserts the resync succeeds on text
    # equality ignoring whitespace: the full tokens use the re-render's token and mask is [1,0,1,0,0,1].
    """The real tokenizer merges adjacent punctuation into whitespace tokens:
    the model's '.\\n\\n' and the re-render's '.\\n' are single, different,
    non-pure-whitespace tokens. Sans-whitespace *text* equality is the check
    that survives this; a per-token pure-whitespace test does not."""
    t0 = _turn([1, 2], [10, 50, 11])  # "...URL.\n\n<tool_call>"
    t1 = _turn([1, 2, 10, 51, 11, 4, 5], [12])  # re-rendered "...URL.\n<tool_call>"
    tokens, mask, logprobs, start = _stitch([t0, t1], tokenizer=_WsTokenizer())
    assert tokens == [1, 2, 10, 51, 11, 4, 5, 12]
    assert mask == [1, 0, 1, 0, 0, 1]


def test_stitch_resync_handles_a_whitespace_run():
    # [FUNCTION DOC] Two single-newline tokens emitted, re-rendered as one double-newline token;
    # asserts the resync realigns with mask [1,0,1,0,1].
    """A two-token whitespace run collapsing to one token still realigns."""
    t0 = _turn([1, 2], [10, 198, 198, 11])
    t1 = _turn([1, 2, 10, 271, 11, 4], [12])
    tokens, mask, logprobs, start = _stitch([t0, t1], tokenizer=_WsTokenizer())
    assert tokens == [1, 2, 10, 271, 11, 4, 12]
    assert mask == [1, 0, 1, 0, 1]


def test_stitch_stops_at_a_rewritten_turn_and_keeps_the_rest():
    # [FUNCTION DOC] Without a tokenizer, a second turn whose history changes a prompt token ends
    # the stitch; asserts only turn one is kept: tokens [1,2,3,10], mask [1], prompt length 3.
    """The harness rewrites a turn's history when it cannot parse the tool call
    it produced. Turns before that are still exactly what the policy emitted,
    so they are kept and the divergent turn ends the episode."""
    t0 = _turn([1, 2, 3], [10])
    t1 = _turn([1, 2, 99, 10, 4], [12])  # 3 -> 99: history was rewritten
    tokens, mask, logprobs, start = _stitch([t0, t1])
    assert tokens == [1, 2, 3, 10]  # turn 0 only
    assert mask == [1] and start == 3


def test_stitch_refuses_when_no_turn_survives():
    # [FUNCTION DOC] A single turn with no output tokens makes the stitch return none.
    """Nothing to train on must be a failure, not an empty sample."""
    t0 = _turn([1, 2], [])  # no output tokens at all
    assert _stitch([t0]) is None


def test_stitch_single_turn():
    # [FUNCTION DOC] One turn: tokens are prompt then output, prompt length 2, mask [1,1,1].
    tokens, mask, logprobs, start = _stitch([_turn([1, 2], [7, 8, 9])])
    assert (tokens, start) == ([1, 2, 7, 8, 9], 2)
    assert mask == [1, 1, 1]


def test_stitch_pads_missing_logprobs_with_zeros():
    # [FUNCTION DOC] A turn with two output tokens but only one log prob (wrong length); asserts the
    # stitched log probs are [0.0, 0.0] (the mismatched list is replaced by zeros).
    tokens, mask, logprobs, _ = _stitch([_turn([1], [7, 8], logprobs=[-0.5])])  # wrong length
    assert logprobs == [0.0, 0.0]


# ---- vision input verification ----


class _FakeImageProcessor:
    # [CLASS DOC] HELPER: Fake image processor with merge size 2. Calling it for n images returns
    # zero pixel values of shape (16n, 8) and a grid of (1, 4, 4) per image, i.e. 16 patches and
    # (after the 2x2 merge) 4 image tokens per image.
    merge_size = 2

    def __call__(self, images, return_tensors):
        # [FUNCTION DOC] HELPER: Builds the mapping with pixel values and image grid tensors for
        # the given images (return type argument ignored). Requires the torch package.
        import torch

        n = len(images)
        # each fake image -> grid (1, 4, 4) = 16 patches -> 4 image tokens
        return {
            "pixel_values": torch.zeros(n * 16, 8),
            "image_grid_thw": torch.tensor([[1, 4, 4]] * n),
        }


def _fake_state():
    # [FUNCTION DOC] HELPER: Builds a fake state holding a processor with the fake image processor
    # and a tokenizer whose token-to-id conversion always returns 777.
    processor = SimpleNamespace(
        image_processor=_FakeImageProcessor(),
        tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda tok: 777),
    )
    return SimpleNamespace(processor=processor)


def test_vision_inputs_verified_against_token_ids():
    # [FUNCTION DOC] Two images and token ids with eight image tokens (four per image): the
    # returned inputs describe two images.
    from PIL import Image

    imgs = [Image.new("RGB", (8, 8)) for _ in range(2)]
    tokens = [777] * 8 + [1, 2]  # 2 images x 4 tokens each
    out = _vision_inputs(_fake_state(), imgs, tokens)
    assert out["image_grid_thw"].shape[0] == 2


def test_vision_inputs_trim_the_screenshot_the_model_never_saw():
    # [FUNCTION DOC] Three images but token ids covering only two; the result describes two images
    # (the trailing unseen screenshot is dropped).
    """The last tool call's screenshot lands after the loop's final model call
    and never enters a prompt, so the trace holds one more image than the ids
    -- an exact prefix, kept; the trailing image, dropped."""
    from PIL import Image

    imgs = [Image.new("RGB", (8, 8)) for _ in range(3)]
    tokens = [777] * 8 + [1]  # ids saw 2 of the 3 images
    out = _vision_inputs(_fake_state(), imgs, tokens)
    assert out["image_grid_thw"].shape[0] == 2


def test_vision_inputs_mismatch_fails_loudly():
    # [FUNCTION DOC] Two images but three image tokens (matches no prefix); raises a value error
    # saying image token mismatch.
    """If the ids' image-token count matches no prefix of the screenshots,
    something upstream disagreed about processing -- training must not
    proceed."""
    from PIL import Image

    with pytest.raises(ValueError, match="image token mismatch"):
        _vision_inputs(_fake_state(), [Image.new("RGB", (8, 8))] * 2, [777] * 3)


def test_vision_inputs_none_without_images():
    # [FUNCTION DOC] No images gives none.
    assert _vision_inputs(_fake_state(), [], [1, 2]) is None


def test_failed_sample_is_shaped_for_the_trainer():
    # [FUNCTION DOC] Marks a fake sample as failed with the reason boom and asserts: removal flag
    # true, empty loss mask and log probs, response length equal to the (zero) mask length, empty
    # response text, and a reward of 0.0 in metadata.
    """A write-off still travels with the batch, so every field the trainer
    indexes has to be present and consistent -- a None here surfaces as a
    TypeError deep in the training step, after the rollout is already spent."""
    from examples.experimental.hud.rollout import _failed

    sample = SimpleNamespace(metadata={}, status=None, remove_sample=False)
    _failed(sample, "boom", state=_fake_state())
    assert sample.remove_sample is True
    assert sample.loss_mask == [] and sample.rollout_log_probs == []
    assert len(sample.loss_mask) == sample.response_length  # what the trainer asserts
    assert sample.response == ""
    assert sample.metadata["reward"] == 0.0


def test_failed_sample_still_exercises_the_vision_tower():
    # [FUNCTION DOC] After marking a sample failed, asserts it carries multimodal inputs for one
    # image and a token list of six ids (four image tokens plus vision start and end), at least
    # as long as the response length, so the vision tower still runs in distributed training.
    """Under FSDP the vision tower's weight all-gathers are collectives, so
    every sample -- write-offs included -- must invoke it exactly once per
    forward. A text-only write-off desynchronizes the ranks' collective
    schedules (observed live as a ~30-minute stall per step)."""
    from examples.experimental.hud.rollout import _failed

    sample = SimpleNamespace(metadata={}, status=None, remove_sample=False)
    _failed(sample, "boom", state=_fake_state())
    assert sample.multimodal_train_inputs is not None
    assert sample.multimodal_train_inputs["image_grid_thw"].shape[0] == 1
    # fake grid (1,4,4), merge 2 -> 4 image tokens + vision_start/end
    assert len(sample.tokens) == 6
    assert len(sample.tokens) >= sample.response_length
