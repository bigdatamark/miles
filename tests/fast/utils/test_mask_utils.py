# [MODULE DOC]
# Tests for the multi-turn loss-mask generator in the mask utilities module, using the Qwen3
# tokenizer-style chat template. Each test loads a pretrained tokenizer by model name (defaults to
# Qwen3-8B; this needs network or cache access to the tokenizer files), generates token ids and
# a parallel loss mask for a conversation, and checks the two have equal length and that the
# number of text segments selected by the mask equals the number of assistant turns. Results are
# also printed. Running the file as a script uses the Qwen3-Coder-30B-A3B-Instruct tokenizer.
from miles.utils.mask_utils import MultiTurnLossMaskGenerator
from miles.utils.processing_utils import load_tokenizer


def test_loss_mask_qwen3_simple(model_name: str = "Qwen/Qwen3-8B"):
    # [FUNCTION DOC] Single-turn case: system, user and assistant messages. Build the generator
    # for the qwen3 tokenizer type, generate ids and mask, assert equal lengths and exactly one
    # selected text, then print the decoded text, ids, mask and selected texts. The model name
    # parameter selects the tokenizer.
    tokenizer = load_tokenizer(model_name)
    mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3")
    messages = [
        {"role": "system", "content": "SYSTEM MESSAGE FOR TESTING ONLY"},
        {"role": "user", "content": "USER CONTENT FOR TESTING ONLY"},
        {"role": "assistant", "content": "ASSISTANT RESPONSE FOR TESTING ONLY"},
    ]
    all_token_ids, all_loss_masks = mask_generator.gen_multi_turn_loss_mask_qwen3(messages)
    assert len(all_token_ids) == len(all_loss_masks), f"{len(all_token_ids)} != {len(all_loss_masks)}"
    selected_texts = mask_generator.get_text_from_loss_mask(all_token_ids, all_loss_masks)
    assert len(selected_texts) == 1, f"Expected 1 text, got {len(selected_texts)}"

    print(f"==== Single Turn Test {model_name} ====")
    print("text = ", [tokenizer.decode(all_token_ids)])
    print("token_ids = ", all_token_ids)
    print("loss_mask = ", all_loss_masks)
    print("selected_texts = ", selected_texts)


def test_loss_mask_qwen3_tools(model_name: str = "Qwen/Qwen3-8B"):
    # [FUNCTION DOC] Multi-turn case with tools: system and user messages, an assistant message with
    # two tool calls, two tool result messages and a final assistant message, plus two tool
    # definitions (a terminal tool and a read-file tool). Generate ids and mask with the tool
    # definitions, assert equal lengths and exactly two selected texts (one per assistant turn),
    # then print the details.
    tokenizer = load_tokenizer(model_name)
    mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3")
    messages = [
        {"role": "system", "content": "SYSTEM MESSAGE FOR TESTING ONLY"},
        {"role": "user", "content": "USER CONTENT FOR TESTING ONLY"},
        {
            "role": "assistant",
            "content": "I WILL CALL terminal",
            "tool_calls": [
                {"function": {"name": "terminal", "arguments": {"command": "ls"}}, "id": "call_0", "type": "function"},
                {"function": {"name": "terminal", "arguments": {"command": "ls"}}, "id": "call_0", "type": "function"},
            ],
        },
        {"role": "tool", "name": "terminal", "content": "LICENSE  README.md  README_zh.md"},
        {"role": "tool", "name": "terminal", "content": "LICENSE  README.md  README_zh.md"},
        {"role": "assistant", "content": "ASSISTANT RESPONSE FOR TESTING ONLY"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "Perform operations from the terminal.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The bash command to execute as `bash -c <command>`",
                        },
                        "description": {
                            "type": "string",
                            "description": "Brief description of the command for the user.",
                        },
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the content of a file given its path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The absolute path to the file to be read.",
                        }
                    },
                    "required": ["file_path"],
                },
            },
        },
    ]

    all_token_ids, all_loss_masks = mask_generator.gen_multi_turn_loss_mask_qwen3(messages, tools)
    assert len(all_token_ids) == len(all_loss_masks), f"{len(all_token_ids)} != {len(all_loss_masks)}"
    selected_texts = mask_generator.get_text_from_loss_mask(all_token_ids, all_loss_masks)
    assert len(selected_texts) == 2, f"Expected 2 texts, got {len(selected_texts)}"

    print(f"==== Multi-turn with Tools Test {model_name} ====")
    print("text = ", [tokenizer.decode(all_token_ids)])
    print("token_ids = ", all_token_ids)
    print("loss_mask = ", all_loss_masks)
    print("selected_texts = ", selected_texts)


if __name__ == "__main__":
    test_loss_mask_qwen3_simple("Qwen/Qwen3-Coder-30B-A3B-Instruct")
    test_loss_mask_qwen3_tools("Qwen/Qwen3-Coder-30B-A3B-Instruct")
