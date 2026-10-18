# [MODULE DOC]
# Test for the multimodal training input extractor in the processing utilities. Registers itself as
# a CPU CI test (estimated 20 s, stage-a-cpu suite, no labels) at import time via the CI register
# call. The extractor takes a processor output mapping and returns only the multimodal tensors
# (such as pixel values and image grid), dropping text ids, attention masks and token type ids; it
# returns none when no multimodal entries remain.
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])

from miles.utils.processing_utils import extract_multimodal_train_inputs


def test_extract_multimodal_train_inputs_drops_qwen3_vl_token_metadata():
    # [FUNCTION DOC] A processor output with input ids, attention mask, token type ids, pixel values
    # and image grid (sentinel objects): the extractor returns a mapping of just the pixel values
    # and image grid (same objects). A processor output with only the three text-side entries
    # returns none.
    pixel_values = object()
    image_grid_thw = object()
    processor_output = {
        "input_ids": [[1, 2, 3]],
        "attention_mask": [[1, 1, 1]],
        "mm_token_type_ids": [[0, 1, 0]],
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }

    assert extract_multimodal_train_inputs(processor_output) == {
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }
    assert (
        extract_multimodal_train_inputs(
            {
                "input_ids": [[1, 2, 3]],
                "attention_mask": [[1, 1, 1]],
                "mm_token_type_ids": [[0, 1, 0]],
            }
        )
        is None
    )
