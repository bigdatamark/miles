# [MODULE DOC] Exact-match reward scoring for the Search-R1 example. Provides answer
# normalization, an exact-match check against a list of golden answers, a state-machine validator
# of the expected response format (alternating think, search, information blocks ending with a
# think and answer block, after the assistant marker), answer and information-block extraction,
# a check whether any retrieved block contains a golden answer, and the combined scoring function
# used by the reward hook.
# Adapt from https://github.com/PeterGriffinJin/Search-R1/blob/ceee7b89655ed52f205b9beb98e1190c3eedcfb0/verl/utils/reward_score/qa_em_format.py
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import re
import string


def normalize_answer(s):
    # [FUNCTION DOC] Normalizes text for comparison. Applied in this order: lowercase, delete ASCII
    # punctuation characters, replace the whole words a, an, the with spaces, collapse whitespace
    # runs to single spaces and trim. Returns the normalized string.
    def remove_articles(text):
        # [FUNCTION DOC] HELPER: Replaces whole-word articles (a, an, the) with a space.
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        # [FUNCTION DOC] HELPER: Collapses any whitespace runs to one space and trims the ends.
        return " ".join(text.split())

    def remove_punc(text):
        # [FUNCTION DOC] HELPER: Deletes every ASCII punctuation character.
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        # [FUNCTION DOC] HELPER: Lowercases the text.
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    # [FUNCTION DOC] Exact-match check. A single string of golden answers is treated as a one-item
    # list. Returns the integer 1 if the normalized prediction equals the normalized form of any
    # golden answer, else 0.
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def is_valid_sequence(text):
    # [FUNCTION DOC] Validates the response format. Returns a pair of validity flag and a message.
    # Looks for the first "<|im_start|>assistant" marker (with trailing whitespace); without it,
    # returns invalid ("Missing assistant marker"). Only text after that marker is examined. Each
    # of the tags think, search, information, answer must have equal numbers of opening and closing
    # occurrences, else invalid with a count message. Then the content is split on the tags and run
    # through a state machine, skipping blank pieces: start -> opening think; closing think ->
    # opening search (then closing search, opening information, closing information, then back to
    # opening think) or opening answer (then closing answer, ending). Any tag not allowed in the
    # current state is invalid. Text between tags is allowed only inside a tag; outside tags only
    # whitespace is allowed. The final state must be the end state, else invalid ("Incomplete
    # sequence"). Otherwise valid.
    # Find the position of "<|im_start|>assistant" with potential whitespace
    assistant_pattern = r"<\|im_start\|>assistant\s*"
    assistant_match = re.search(assistant_pattern, text)

    if not assistant_match:
        return False, "Missing assistant marker"

    # Extract the content after the assistant marker
    start_pos = assistant_match.end()
    content = text[start_pos:]

    # Check for balanced tags
    tags_to_check = ["think", "search", "information", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"

    # Now check for proper sequence pattern and no extraneous content

    # 1. First split the content by any tags we recognize
    split_pattern = r"(</?(?:think|search|information|answer)>)"
    parts = re.split(split_pattern, content)

    # 2. Keep track of the current position in the expected sequence
    state = "start"  # start -> think -> search -> information -> think -> ... -> answer -> end

    # 3. Check each part
    for _i, part in enumerate(parts):
        # Skip empty parts
        if not part.strip():
            continue

        # Check if this is a tag
        if re.match(r"</?(?:think|search|information|answer)>", part):
            # This is a tag, check if it's valid in the current state
            if part == "<think>" and state in ["start", "information"]:
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<search>" and state == "after_think":
                state = "in_search"
            elif part == "</search>" and state == "in_search":
                state = "after_search"
            elif part == "<information>" and state == "after_search":
                state = "in_information"
            elif part == "</information>" and state == "in_information":
                state = "information"
            elif part == "<answer>" and state == "after_think":
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            # This is content, check if it's valid in the current state
            if state in ["in_think", "in_search", "in_information", "in_answer"]:
                # Content is allowed inside tags
                pass
            elif state in ["start", "after_think", "after_search", "information"]:
                # Only whitespace is allowed between tags
                if part.strip():
                    return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
            else:
                return False, f"Unexpected content in state {state}"

    # Check final state
    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"

    return True, "Valid sequence format"


def extract_solution(solution_str):
    # [FUNCTION DOC] Returns the trimmed content of the LAST answer-tagged span (multi-line) in the
    # text, but only if there are at least two such spans; with zero or one span returns none. The
    # two-span requirement is because the prompt text, which is part of the scored string, contains
    # one example answer span.
    # KNOWN ISSUE: the docstring speaks of extracting an equation, which does not match the code.
    """Extract the equation from the solution string."""

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def extract_information_blocks(text: str) -> list[str]:
    # [FUNCTION DOC] HELPER: Returns the trimmed contents of every information-tagged span
    # (multi-line, shortest match), in order.
    pattern = r"<information>(.*?)</information>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def is_retrieval_correct(text: str, golden_answers: list[str]) -> list[str]:
    # [FUNCTION DOC] HELPER: Returns true if any information block, normalized, contains any
    # normalized golden answer as a substring; otherwise false.
    # KNOWN ISSUE: the return annotation says list of strings but a boolean is returned.
    seqs = extract_information_blocks(text)
    for seq in seqs:
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(seq):
                return True
    return False


def compute_score_em(
    solution_str,
    ground_truth,
    method="strict",
    structure_format_score=0,
    final_format_score=0,
    retrieval_score=0,
    format_score=0,
    score=1.0,
):
    # [FUNCTION DOC] Scores one solution string (prompt plus response). The ground truth is a
    # mapping whose "target" entry holds the golden answers. Computes format validity, retrieval
    # correctness (only evaluated when the format is valid), and the extracted answer. With
    # probability 1/64 prints the golden answers, extracted answer and full text to standard
    # output (random side effect). Result: no extracted answer -> structure score plus retrieval
    # score if format valid and retrieval correct, structure score if only format valid, else 0.
    # Extracted answer matching exactly -> the full score if format valid, else the full score minus
    # the structure score. Not matching -> as for no answer if format valid, else the final format
    # score. With default parameters all auxiliary scores are zero.
    # KNOWN ISSUE: the method and format-score parameters are never used (callers passing a format
    # score get no effect), and the inline comments quoting 0.3, 0.2, 0.8 and 0.1 only hold for
    # non-default parameter values, since the defaults for the auxiliary scores are zero.
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    is_valid_format, _ = is_valid_sequence(solution_str)
    retrieval_correct = False
    if is_valid_format:
        retrieval_correct = is_retrieval_correct(solution_str, ground_truth["target"])
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        if is_valid_format:
            if retrieval_correct:
                return structure_format_score + retrieval_score  # 0.3
            else:
                return structure_format_score  # 0.2
        else:
            return 0
    else:
        if em_check(answer, ground_truth["target"]):
            if is_valid_format:
                return score  # 1
            else:
                return score - structure_format_score  # 0.8
        elif is_valid_format:
            if retrieval_correct:
                return structure_format_score + retrieval_score  # 0.3
            else:
                return structure_format_score  # 0.2
        else:
            return final_format_score  # 0.1
