# [MODULE DOC]
# Rule-based reward for GPQA-style multiple-choice questions. Extracts the option letter the model
# chose from its response (after removing any reasoning section) and compares it with the correct
# letter derived from the label and sample metadata (choices, valid letters, correct letter,
# answer text); falls back to matching the answer text inside the response. Module constant:
# default valid letters are the first eight uppercase letters A to H.
import re
import string
from collections.abc import Iterable

DEFAULT_VALID_LETTERS = list(string.ascii_uppercase[:8])


def _strip_chain_of_thought(text: str) -> str:
    # [FUNCTION DOC] HELPER: Return the text after the last think-close tag if present; empty
    # string for empty or null input; otherwise the text unchanged.
    if not text:
        return ""

    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1]

    return text


def _normalize_text(text: str) -> str:
    # [FUNCTION DOC] HELPER: Lower-case, replace each run of characters other than lower-case
    # letters and digits with one space, and trim the ends.
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _extract_letter_from_response(response: str, valid_letters: Iterable[str]) -> str | None:
    # [FUNCTION DOC] HELPER: Return the chosen option letter (upper-case) or none. Empty response
    # gives none. After stripping the reasoning section, try three case-insensitive patterns in
    # order: "answer / option / choice" followed by an optional "is" or colon and a letter; a
    # letter followed by "is correct"; "final answer / final option" followed by an optional "is"
    # or colon and a letter; the first match whose letter (upper-cased) is valid wins (only the
    # first match of each pattern is examined). Fallback: the last standalone capital letter in the
    # text that is valid. Valid letters are compared in upper case.
    # KNOWN ISSUE: because the patterns are case-insensitive and the letter group is not
    # restricted by the valid set until after matching, a phrase like "answer is a ..." matches
    # the article as an option letter; also the second pattern is shadowed by looser matches.
    """
    Best-effort extraction of the selected option letter from the model response.
    """
    if not response:
        return None

    text = _strip_chain_of_thought(response)
    patterns = [
        r"(?:answer|option|choice)\s*(?:is|:)?\s*([A-Z])",
        r"([A-Z])\s*(?:is\s*(?:the)?\s*correct)",
        r"final\s*(?:answer|option)\s*(?:is|:)?\s*([A-Z])",
    ]

    valid_letters = {letter.upper() for letter in valid_letters}
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            if letter in valid_letters:
                return letter

    # Fallback: last standalone capital letter that is valid.
    candidates = re.findall(r"\b([A-Z])\b", text)
    for letter in reversed(candidates):
        letter = letter.upper()
        if letter in valid_letters:
            return letter

    return None


def compute_gpqa_reward(response: str, label, metadata: dict | None = None) -> float:
    # [FUNCTION DOC]
    # Score a response 1.0 or 0.0. Null response gives 0.0. The optional metadata mapping may hold:
    # choices (list or mapping of option texts), valid letters, correct letter, correct answer,
    # answer text. Side effect: may add a correct answer entry to the metadata when the label text
    # matches a choice.
    # Steps: 1. Choices become a list. 2. Valid letters: from metadata (upper-cased), else the first
    # N uppercase letters for N choices, else A to H. 3. Correct letter: metadata value if a
    # string (stripped, upper-cased). A string label of one valid letter sets it if still unset;
    # a numeric label is an index into the valid letters (out of range ignored). If still unset
    # and there are choices and a label text, find the choice whose normalized text equals the
    # normalized label and use its letter. 4. Extract the response letter; if both letters exist,
    # return 1.0 on equality else 0.0. 5. Otherwise build candidate answer texts: the choice of
    # the correct letter, the metadata correct answer and answer text, and the label text; if any
    # normalized non-empty candidate is a substring of the normalized response (reasoning removed)
    # return 1.0. 6. If a letter was extracted, no correct letter is known but a label text exists,
    # return 1.0 if the letter equals the upper-cased label text. 7. Else 0.0.
    """Rule-based scorer for GPQA-style multiple-choice evaluation."""
    if response is None:
        return 0.0

    metadata = metadata or {}

    choices = metadata.get("choices")
    if isinstance(choices, dict):
        choices = list(choices.values())
    elif choices is not None:
        choices = list(choices)

    valid_letters = metadata.get("valid_letters")
    if valid_letters:
        valid_letters = [str(letter).upper() for letter in valid_letters]
    elif choices:
        valid_letters = list(string.ascii_uppercase[: len(choices)])
    else:
        valid_letters = DEFAULT_VALID_LETTERS

    correct_letter = metadata.get("correct_letter")
    if isinstance(correct_letter, str):
        correct_letter = correct_letter.strip().upper()
    else:
        correct_letter = None

    label_text = None
    if isinstance(label, str):
        label_text = label.strip()
        if len(label_text) == 1 and label_text.upper() in valid_letters and not correct_letter:
            correct_letter = label_text.upper()
    elif isinstance(label, (int, float)):
        idx = int(label)
        if 0 <= idx < len(valid_letters):
            correct_letter = valid_letters[idx]

    if not correct_letter and choices and label_text:
        normalized_label = _normalize_text(label_text)
        for idx, choice in enumerate(choices):
            if _normalize_text(str(choice)) == normalized_label:
                correct_letter = valid_letters[idx]
                metadata.setdefault("correct_answer", choice)
                break

    extracted_letter = _extract_letter_from_response(response, valid_letters)
    if extracted_letter and correct_letter:
        return 1.0 if extracted_letter == correct_letter else 0.0

    candidate_answers = []
    if correct_letter and choices:
        try:
            idx = valid_letters.index(correct_letter)
        except ValueError:
            idx = None
        if idx is not None and idx < len(choices):
            candidate_answers.append(str(choices[idx]))

    for key in ("correct_answer", "answer_text"):
        value = metadata.get(key)
        if value:
            candidate_answers.append(str(value))

    if label_text:
        candidate_answers.append(label_text)

    normalized_targets = {_normalize_text(text) for text in candidate_answers if text}
    normalized_response = _normalize_text(_strip_chain_of_thought(response))
    for target in normalized_targets:
        if target and target in normalized_response:
            return 1.0

    if extracted_letter and not correct_letter and label_text:
        return 1.0 if extracted_letter == label_text.strip().upper() else 0.0

    return 0.0
