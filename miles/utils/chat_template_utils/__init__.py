# [PACKAGE DOC] Chat template handling for agentic workflows, aimed at keeping the tokens that
# training sees identical to those the inference server produced. Contents: message_matcher_hub
# (dependency-free policies deciding whether a replayed message equals a stored one, plus
# append-only validation), template (loading and rendering chat templates in the inference
# server's style), deepseek (bridge to the DeepSeek official encoders), inkling and
# inkling_parser (Inkling model support), tito_tokenizer (token-in-token-out tokenizers for
# incremental tokenization) and token_seq_comparator (token sequence diffing); templates holds
# bundled template assets. This file re-exports the main public names; import order matters
# because of the dependency chain noted below.

"""Chat template utilities for agentic-workflow token consistency."""

# Import order matters: message_matcher_hub is the dependency-free base,
# template re-exports its two moved names, and tito_tokenizer builds on both.
from miles.utils.chat_template_utils.message_matcher_hub import (
    assert_messages_append_only_with_allowed_role,
    strict_message_matches,
)
from miles.utils.chat_template_utils.template import (
    apply_chat_template,
    apply_chat_template_from_str,
    extract_tool_dicts,
    load_hf_chat_template,
    normalize_tool_arguments,
)
from miles.utils.chat_template_utils.tito_tokenizer import (
    TEMPLATE_DIR,
    TITOTokenizer,
    TITOTokenizerType,
    get_tito_tokenizer,
    resolve_fixed_chat_template,
    resolve_reasoning_and_tool_call_parser,
)
from miles.utils.chat_template_utils.token_seq_comparator import Mismatch, MismatchType, TokenSeqComparator

__all__ = [
    "TITOTokenizer",
    "TITOTokenizerType",
    "get_tito_tokenizer",
    "TEMPLATE_DIR",
    "resolve_fixed_chat_template",
    "resolve_reasoning_and_tool_call_parser",
    "load_hf_chat_template",
    "apply_chat_template",
    "apply_chat_template_from_str",
    "assert_messages_append_only_with_allowed_role",
    "strict_message_matches",
    "extract_tool_dicts",
    "normalize_tool_arguments",
    "Mismatch",
    "TokenSeqComparator",
    "MismatchType",
]
