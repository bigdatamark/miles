# [MODULE DOC]
# Rule-based math reward graders in the DeepScaleR style: take the final answer from the last
# boxed expression of the model output (after stripping the reasoning segment) and compare it
# with the label using string-normalized and symbolic equivalence. Rewards are integers 0 or 1.
from .math_utils import extract_answer, grade_answer_mathd, grade_answer_sympy


def _grade_boxed_solution(model_solution, label):
    # [FUNCTION DOC] HELPER: Grade a solution text against a label; returns 1 if correct else 0.
    # Extract the boxed answer from the solution (none gives 0). An empty-string label gives 0.
    # The label must be a string, float or int (assertion error otherwise) and is converted to
    # text; if it contains a boxed expression, the boxed content replaces it (a boxed label with
    # no extractable content gives 0). Correct when the string-normalizing grader or the symbolic
    # grader accepts the model answer against the ground truth.
    model_answer = extract_answer(model_solution)
    if model_answer is None:
        return 0
    if label == "":
        return 0

    # Convert single answer to list for uniform processing
    assert isinstance(label, (str, float, int))
    ground_truths = [label]

    # Process each ground truth
    processed_ground_truths = []
    for truth in ground_truths:
        truth = str(truth)
        if "\\boxed" in truth:
            processed_truth = extract_answer(truth)
            if processed_truth is not None:
                processed_ground_truths.append(processed_truth)
        else:
            processed_ground_truths.append(truth)

    if not processed_ground_truths:
        return 0

    # Check against all possible correct answers
    for ground_truth in processed_ground_truths:
        is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
        if is_correct:
            return 1

    return 0


def get_deepscaler_rule_based_reward(response, label):
    # [FUNCTION DOC]
    # Reward 0/1 for a response with a reasoning section. Looks for the first of three end-of-
    # reasoning markers (tagged open-response-separator, think close tag, and a response header)
    # present in the response, in that priority order, and grades the text after the last
    # occurrence of that marker. If no marker is present the reward is 0.
    # markers that end the reasoning segment; Kimi K-series uses the tagged <|open|>response<|sep|>
    for closer in ("<|open|>response<|sep|>", "</think>", "###Response"):
        if closer in response:
            return _grade_boxed_solution(response.split(closer)[-1], label)
    return 0


def get_gemma_math_reward(response, label):
    # [FUNCTION DOC]
    # Reward 0/1 for Gemma-style output: if the thinking-close channel marker is present, keep
    # only the text after its last occurrence; then grade the boxed answer. Unlike the deepscaler
    # variant, a response without any marker is still graded in full.
    # Gemma-4 closes thinking with <channel|>; grade text after the last one.
    if "<channel|>" in response:
        response = response.split("<channel|>")[-1]
    return _grade_boxed_solution(response, label)
