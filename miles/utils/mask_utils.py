# [MODULE DOC]
# Loss mask generation for multi-turn chat data. The generator tokenizes messages with the chat
# template and marks assistant tokens (after the generation prompt prefix) with one and everything
# else with zero; variants exist for qwen, qwen3 and distilled qwen templates. Template overhead
# (system message length and generation prompt length) is probed once from the tokenizer.
from transformers import AutoTokenizer


def get_response_lengths(loss_masks: list[list[int]]) -> list[int]:
    # [FUNCTION DOC] For each mask returns the length from its first one to the end (zero for masks
    # without any one).
    # return the lengths starting from the first occurrence of 1 to the end of each loss mask
    return [len(mask[mask.index(1) :]) if 1 in mask else 0 for mask in loss_masks]


class MultiTurnLossMaskGenerator:
    # [CLASS DOC] Builds token ids and loss masks for message lists. Holds the tokenizer, the chat
    # template flavor (qwen, qwen3, distill_qwen), and two probed lengths: tokens a template adds
    # before a non-first message body (system message length) and tokens of the generation prompt.
    def __init__(self, tokenizer: AutoTokenizer, tokenizer_type: str = "qwen"):
        # [FUNCTION DOC] Stores the tokenizer, probes the template lengths immediately (tokenizer
        # calls), then stores the flavor name (unsupported names fail later in mask generation).
        self.tokenizer = tokenizer
        self.system_message_length, self.gen_token_length = self.get_system_message_length()
        self.tokenizer_type = tokenizer_type

    def get_response_lengths(self, loss_masks: list[list[int]]) -> list[int]:
        # [FUNCTION DOC] Method wrapper over the module-level function of the same name.
        return get_response_lengths(loss_masks)

    def find_all_sublist_indices(self, main_list, sublist):
        # [FUNCTION DOC] Start indices of every (possibly overlapping) occurrence of the sublist in
        # the main list, ascending; empty if none.
        sublist_len = len(sublist)
        indices = []
        for i in range(len(main_list) - sublist_len + 1):
            if main_list[i : i + sublist_len] == sublist:
                indices.append(i)
        return indices

    def get_system_message_length(self) -> tuple[int, int]:
        # [FUNCTION DOC] Probes the chat template with two identical user messages containing a test
        # string. Finds the two occurrences of the raw test token ids inside the templated token ids
        # (unpacking fails with a value error unless there are exactly two). The template tail length
        # is total minus raw minus second index. The generation prompt length is the token count with
        # the generation prompt added minus without. Returns (system message length, generation
        # prompt length) where the first is the first index minus the per-message overhead
        # ((second minus first index) minus tail length minus raw length).
        # KNOWN ISSUE: the generation length subtracts a token count of the tokenized chat template
        # output from the length of the apply result, which assumes that call returns a flat id list
        # (not a mapping) despite the dict-return flag being false; and the name says system message
        # length while the value is the per-message template prefix length.
        test_string = "FOR TESTING ONLY"
        test_messages = [
            {"role": "user", "content": test_string},
            {"role": "user", "content": test_string},
        ]
        raw_token_ids = self.tokenizer(test_string, add_special_tokens=False)["input_ids"]
        chat_template_token = self.tokenizer.apply_chat_template(
            test_messages, add_special_tokens=False, tokenize=False
        )
        chat_template_token_ids = self.tokenizer(chat_template_token, add_special_tokens=False)["input_ids"]
        idx_1, idx_2 = self.find_all_sublist_indices(chat_template_token_ids, raw_token_ids)
        end_interval = len(chat_template_token_ids) - len(raw_token_ids) - idx_2
        gen_token_length = len(
            self.tokenizer.apply_chat_template(
                test_messages, add_special_tokens=False, tokenize=True, return_dict=False, add_generation_prompt=True
            )
        ) - len(chat_template_token_ids)

        system_message_length = idx_1 - ((idx_2 - idx_1) - end_interval - len(raw_token_ids))
        return system_message_length, gen_token_length

    def gen_multi_turn_loss_mask_qwen(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        # [FUNCTION DOC] Qwen flavor. Tokenizes each message alone with the chat template (tools only
        # for the first message). For non-system messages after the first, drops the leading
        # system-message-length tokens (the template's default preamble). Assistant messages get
        # zeros for the generation prompt tokens and ones for the rest; others all zeros; a message
        # whose step loss mask field is present and not one gets all zeros. Returns (token ids, loss
        # mask) concatenated over messages.
        all_loss_masks = []
        all_token_ids = []

        for i, message in enumerate(messages):
            if i == 0:
                message_ids = self.tokenizer.apply_chat_template(
                    [message], tokenize=True, return_dict=False, tools=tools
                )
            else:
                message_ids = self.tokenizer.apply_chat_template([message], tokenize=True, return_dict=False)

            if message["role"] != "system" and i > 0:
                message_ids = message_ids[self.system_message_length :]

            if message["role"] == "assistant":
                loss_mask = [0] * self.gen_token_length + [1] * (len(message_ids) - self.gen_token_length)
            else:
                loss_mask = [0] * len(message_ids)

            if message.get("step_loss_mask", 1) != 1:
                loss_mask = [0] * len(message_ids)

            all_loss_masks.extend(loss_mask)
            all_token_ids.extend(message_ids)

        return all_token_ids, all_loss_masks

    def gen_multi_turn_loss_mask_qwen3(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        # [FUNCTION DOC] Qwen3 flavor. Like the qwen flavor, but each message is tokenized together
        # with a fixed dummy user message so templates that depend on neighbors render correctly: the
        # first message is templated followed by the dummy and the dummy's token count is cut from
        # the end; later messages are templated after the dummy and the dummy's token count is cut
        # from the start. Then the same preamble drop, masking and step loss rule apply. Returns
        # (token ids, loss mask).
        all_loss_masks = []
        all_token_ids = []

        prefix_message = {"role": "user", "content": "FOR CALCULATING LOSS MASK ONLY"}
        prefix_token_ids = self.tokenizer.apply_chat_template([prefix_message], tokenize=True, return_dict=False)

        for i, message in enumerate(messages):
            if i == 0:
                tailed_message_ids = self.tokenizer.apply_chat_template(
                    [message, prefix_message], tokenize=True, return_dict=False, tools=tools
                )
                message_ids = tailed_message_ids[: -len(prefix_token_ids)]
            else:
                prefixed_message_ids = self.tokenizer.apply_chat_template(
                    [prefix_message, message], tokenize=True, return_dict=False
                )
                message_ids = prefixed_message_ids[len(prefix_token_ids) :]

            if message["role"] != "system" and i > 0:
                message_ids = message_ids[self.system_message_length :]

            if message["role"] == "assistant":
                loss_mask = [0] * self.gen_token_length + [1] * (len(message_ids) - self.gen_token_length)
            else:
                loss_mask = [0] * len(message_ids)

            if message.get("step_loss_mask", 1) != 1:
                loss_mask = [0] * len(message_ids)

            all_loss_masks.extend(loss_mask)
            all_token_ids.extend(message_ids)

        return all_token_ids, all_loss_masks

    def gen_multi_turn_loss_mask_distill_qwen(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        # [FUNCTION DOC] Distilled qwen flavor (single prompt, single response). Prompt text is the
        # template applied to only the first message with the generation prompt appended; the response
        # is the content of the last message. Both are tokenized without special tokens and
        # concatenated; mask is zeros over the prompt and ones over the response, all zeros if the
        # last message's step loss mask field is present and not one. Middle messages are ignored.
        prompt = self.tokenizer.apply_chat_template(
            messages[:1], tokenize=False, add_generation_prompt=True, tools=tools
        )
        response = messages[-1]["content"]
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_tokens = self.tokenizer(response, add_special_tokens=False)["input_ids"]

        response_length = len(response_tokens)
        token_ids = prompt_tokens + response_tokens
        loss_mask = [0] * len(prompt_tokens) + [1] * response_length

        if messages[-1].get("step_loss_mask", 1) != 1:
            loss_mask = [0] * len(token_ids)
        return token_ids, loss_mask

    def get_loss_mask(self, messages: list[dict], tools: list[dict] = None) -> tuple[list[int], list[int]]:
        # [FUNCTION DOC] Dispatches on the flavor. Qwen: uses the distilled variant if the tokenizer's
        # added vocabulary contains the full-width-bar assistant marker, else the qwen variant. Qwen3
        # and distill_qwen use their variants. Any other flavor raises a value error. Returns (token
        # ids, loss mask).
        if self.tokenizer_type == "qwen":
            if "<｜Assistant｜>" in self.tokenizer.get_added_vocab():
                return self.gen_multi_turn_loss_mask_distill_qwen(messages, tools)

            return self.gen_multi_turn_loss_mask_qwen(messages, tools)
        elif self.tokenizer_type == "qwen3":
            return self.gen_multi_turn_loss_mask_qwen3(messages, tools)
        elif self.tokenizer_type == "distill_qwen":
            return self.gen_multi_turn_loss_mask_distill_qwen(messages, tools)
        else:
            raise ValueError(f"Unsupported tokenizer type: {self.tokenizer_type}")

    def get_loss_mask_with_multimodal_alignment(
        self, messages: list[dict], input_ids: list[int], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        # [FUNCTION DOC] Computes a mask for multimodal inputs whose real token ids are longer than
        # the text-only tokenization. Flattens list-valued message contents into one space-joined
        # string of text parts (dict parts of type text and bare strings; other parts dropped), computes
        # the text-only loss mask, asserts the given input ids are not shorter than it, and left-pads
        # the mask with zeros to the input length (assumes the extra tokens come before the text
        # tokens). Returns (the given input ids unchanged, mask).
        text = []
        for msg in messages:
            if isinstance(msg.get("content"), list):
                text_parts = []
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        text_parts.append(item)
                text.append({"role": msg["role"], "content": " ".join(text_parts)})
            else:
                text.append(msg)

        _, loss_mask_text = self.get_loss_mask(text, tools=tools)

        diff = len(input_ids) - len(loss_mask_text)
        assert diff >= 0, (
            f"input_ids (length={len(input_ids)}) is shorter than text loss_mask (length={len(loss_mask_text)}) "
            f"Please check if processor and tokenizer tokenization are consistent."
        )
        loss_mask = [0] * diff + loss_mask_text

        return input_ids, loss_mask

    def get_text_from_loss_mask(self, token_ids: list[int], loss_masks: list[int]) -> list[str]:
        # [FUNCTION DOC] Decodes each maximal run of tokens whose mask is one into a text; returns the
        # list of decoded runs in order (empty if none).
        selected_texts = []
        current_tokens = []

        for idx, mask in enumerate(loss_masks):
            if mask == 1:
                current_tokens.append(token_ids[idx])
            elif current_tokens:
                selected_texts.append(self.tokenizer.decode(current_tokens))
                current_tokens = []

        if current_tokens:
            selected_texts.append(self.tokenizer.decode(current_tokens))

        return selected_texts
