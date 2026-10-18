# [MODULE DOC] Tests of tool-response tokenization for the generate hub. Loads real Hugging Face
# tokenizers by model name (network or cache access needed; remote code is trusted) and checks that
# tokenizing tool response messages yields the same text as the incremental difference of rendering
# a dummy conversation with and without the tool messages through the chat template.
# Module constants: the list of models under test, a documentation-only list of models known to
# fail round trips under newer tokenizer library versions (with reasons), a list of models that
# support only a single tool call (currently empty), and two sample tool response messages.
import pytest

from miles.rollout.generate_utils.tool_call_utils import _DUMMY_USER, _build_dummy_assistant, tokenize_tool_responses
from miles.utils.processing_utils import load_tokenizer

TOOL_CALL_TEST_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3-Coder-Next",
    # "meta-llama/Llama-3.2-1B-Instruct",  # Skipped: gated repo, requires HF_TOKEN in CI
    "mistralai/Mistral-7B-Instruct-v0.3",
    "MiniMaxAI/MiniMax-M2.5",
    "MiniMaxAI/MiniMax-M2.7",
    "internlm/internlm3-8b-instruct",
    "zai-org/GLM-4.7-Flash",
    # "stepfun-ai/Step-3.5-Flash",  # Disabled: config num_hidden_layers/layer_types mismatch breaks huggingface_hub strict validation
    "moonshotai/Kimi-K2-Instruct",
    "moonshotai/Kimi-K2.5",
    "XiaomiMiMo/MiMo-7B-RL",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
]

# Models that fail decode round-trip under transformers>=5.x due to upstream tokenizer issues.
# These are excluded from TOOL_CALL_TEST_MODELS but listed here for tracking.
# - DeepSeek-V3, step3: transformers v5 unified LlamaTokenizer overwrites their ByteLevel
#   pre_tokenizer/decoder with Metaspace, causing decode(encode(text)) != text.
#   See https://github.com/huggingface/transformers/issues/43066
# - DeepSeek-V3.1: its tool-call chat template concatenates function.arguments as a string,
#   but our dummy tool-call shape provides a dict, raising TypeError before the round-trip check.
# - glm-4-9b-chat: v5 removed the legacy _decode special-token segmentation, exposing a bug in
#   the model's custom convert_tokens_to_string (doesn't handle str-type special tokens).
TOOL_CALL_KNOWN_FAILURES = [
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-V3.1",
    "stepfun-ai/step3",
    "THUDM/glm-4-9b-chat",
]

SINGLE_TOOL_CALL_ONLY_MODELS = [
    # "meta-llama/Llama-3.2-1B-Instruct",  # Skipped: gated repo
]

SAMPLE_TOOL_RESPONSES = [
    {
        "role": "tool",
        "tool_call_id": "call00000",
        "content": '{"year": 2026}',
        "name": "get_year",
    },
    {
        "role": "tool",
        "tool_call_id": "call00001",
        "content": '{"temperature": 25}',
        "name": "get_temperature",
    },
]


class TestTokenizeToolResponses:
    # [CLASS DOC] Tests of the tool-response tokenization function.
    @pytest.mark.parametrize("model_name", ["Qwen/Qwen3-0.6B"])
    def test_snapshot(self, model_name):
        # [FUNCTION DOC] Tokenizes both sample tool responses with the Qwen3 tokenizer, decodes
        # them, and asserts the exact expected text: a user turn wrapping each response in tool
        # response tags, closed by an end-of-turn marker, followed by an opening assistant turn.
        tokenizer = load_tokenizer(model_name, trust_remote_code=True)
        token_ids = tokenize_tool_responses(SAMPLE_TOOL_RESPONSES, tokenizer)
        decoded = tokenizer.decode(token_ids)

        assert decoded == (
            "<|im_start|>user\n"
            "<tool_response>\n"
            '{"year": 2026}\n'
            "</tool_response>\n"
            "<tool_response>\n"
            '{"temperature": 25}\n'
            "</tool_response><|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    @pytest.mark.parametrize("num_tools", [1, 2])
    @pytest.mark.parametrize("model_name", TOOL_CALL_TEST_MODELS)
    def test_tokenize_tool_responses(self, model_name, num_tools):
        # [FUNCTION DOC] Parametrized over every model in the test list and one or two tool
        # responses. Skips combinations of two responses with single-tool-only models. Tokenizes
        # the first num_tools responses, decodes them, and computes the expected text as the
        # suffix added when the responses are appended to the dummy user/assistant conversation
        # (rendered with the generation prompt) relative to the conversation alone (rendered
        # without it). Asserts the two strings are equal, reporting the model name on failure.
        if num_tools > 1 and model_name in SINGLE_TOOL_CALL_ONLY_MODELS:
            pytest.skip(f"{model_name} only supports single tool call")

        tokenizer = load_tokenizer(model_name, trust_remote_code=True)

        tool_responses = SAMPLE_TOOL_RESPONSES[:num_tools]
        assert len(tool_responses) == num_tools

        actual_token_ids = tokenize_tool_responses(tool_responses, tokenizer)
        actual_str = tokenizer.decode(actual_token_ids)

        dummy_assistant = _build_dummy_assistant(tool_responses)
        base_messages = [_DUMMY_USER, dummy_assistant]
        expected_str = self._compute_chat_template_diff(base_messages, tool_responses, tokenizer)

        assert actual_str == expected_str, f"{model_name=}"

    @staticmethod
    def _compute_chat_template_diff(base_messages, extra_messages, tokenizer) -> str:
        # [FUNCTION DOC] HELPER: Static method. Renders the base plus extra messages as text with
        # the generation prompt appended, renders the base messages alone without it, and returns
        # the part of the first string after the length of the second (assumes the second is a
        # prefix of the first; not checked).
        text_with = tokenizer.apply_chat_template(
            base_messages + extra_messages, tokenize=False, add_generation_prompt=True
        )
        text_without = tokenizer.apply_chat_template(base_messages, tokenize=False, add_generation_prompt=False)
        return text_with[len(text_without) :]
