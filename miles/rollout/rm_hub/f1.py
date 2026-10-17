# [MODULE DOC]
# Token-overlap F1 scoring for short text answers (SQuAD / HotpotQA style): normalizes both
# strings and computes F1, precision and recall over word multiset overlap.
import re
import string
from collections import Counter


def normalize_answer(s):
    # [FUNCTION DOC]
    # Normalize an answer string: lower-case, remove all ASCII punctuation characters, remove the
    # standalone articles a, an and the, then collapse whitespace runs to single spaces and trim.
    # Returns the normalized string.

    def remove_articles(text):
        # [FUNCTION DOC] HELPER: Replace whole-word a, an, the (case-sensitive) with a space.
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        # [FUNCTION DOC] HELPER: Collapse any whitespace run to one space and trim the ends.
        return " ".join(text.split())

    def remove_punc(text):
        # [FUNCTION DOC] HELPER: Delete every ASCII punctuation character.
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        # [FUNCTION DOC] HELPER: Lower-case the text.
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    # [FUNCTION DOC]
    # Compute (F1, precision, recall) between a predicted and a reference answer string; returns
    # the triple of zeros if the prediction is null, there is no word overlap, or a yes/no/noanswer
    # special answer on either side does not exactly equal the other side after normalization.
    # Otherwise tokens are whitespace-split normalized words; overlap is the size of the multiset
    # intersection; precision is overlap over prediction words, recall is overlap over reference
    # words, and F1 is their harmonic mean. Pure function.
    ZERO_METRIC = (0, 0, 0)

    if prediction is None:
        return ZERO_METRIC

    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall
