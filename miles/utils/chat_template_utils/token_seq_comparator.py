# [MODULE DOC] Compares two token-id sequences (an expected, template-rendered one and an actual,
# pretokenized one) after splitting them at special-token boundaries, and classifies differences.
# Defines Segment (a run of token ids, special or content), MismatchType (four kinds of
# difference), Mismatch (one reported difference, convertible to a plain mapping), the
# TokenSeqComparator class, and a module-private helper that trims trailing stop tokens.
"""TokenSeqComparator: segment token IDs by special-token boundaries and compare sequences."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass
class Segment:
    # [CLASS DOC] Data record: a list of token ids (default empty) and a flag saying whether the
    # run is a single special token (default false, meaning a content run).
    """A contiguous run of token IDs — either a special token or a content segment."""

    token_ids: list[int] = field(default_factory=list)
    is_special: bool = False


class MismatchType(Enum):
    # [CLASS DOC] Enumeration of difference kinds; each value is a lowercase string. Special token
    # count (structure differs, reported once with segment index -1), special token type (same
    # position, different special id), non-assistant text (should never differ; indicates a bug in
    # the incremental tokenization), assistant text (expected, non-severe).
    # Segment count or structure (special/content pattern) differs between
    # expected and actual.  When this happens, segments can't be aligned so
    # no per-segment comparison is possible.
    SPECIAL_TOKEN_COUNT = "special_token_count"

    # A special-token segment has the same position in both sequences but
    # contains a different token ID.
    SPECIAL_TOKEN_TYPE = "special_token_type"

    # Non-assistant content (user, system, tool, etc.) differs.  This indicates
    # a bug in the TITO algorithm — these regions should match exactly.
    NON_ASSISTANT_TEXT = "non_assistant_text"

    # Assistant content differs.  Expected and non-severe: assistant tokens
    # are inherited directly from the pretokenized prefix across turns,
    # so they may not match the chat template's canonical tokenization.
    ASSISTANT_TEXT = "assistant_text"


@dataclass
class Mismatch:
    # [CLASS DOC] Data record for one difference: its kind, the segment index it occurred at (-1
    # for a whole-sequence structural difference), decoded expected text, decoded actual text, and
    # an optional detail message (default empty).
    """A single difference found between two token sequences."""

    type: MismatchType
    segment_index: int
    expected_text: str
    actual_text: str
    detail: str = ""

    def to_dict(self) -> dict:
        # [FUNCTION DOC] Returns a plain mapping with the keys type (the kind's string value),
        # segment_index, expected_text, actual_text and detail, for serialization.
        return {
            "type": self.type.value,
            "segment_index": self.segment_index,
            "expected_text": self.expected_text,
            "actual_text": self.actual_text,
            "detail": self.detail,
        }


class TokenSeqComparator:
    # [CLASS DOC] Stateless-after-construction comparator. Holds the tokenizer, the set of special
    # token ids that delimit segments (given, or auto-detected from the tokenizer), the assistant
    # start string used to classify content segments as assistant text, and an optional default
    # set of trailing token ids to trim from both sequences before comparing. Compare flow: trim
    # tails, segment both sequences, check structure (count and special/content pattern), then
    # compare aligned segments one by one.
    """Segment token sequences by special tokens and compare them.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerBase
    special_token_ids : set[int] | None
        Token IDs that mark segment boundaries.  Default None auto-detects
        them (see :meth:`collect_special_ids`).  Pass an explicit set to
        override — needed when the tokenizer flags a token ``special=True``
        even though it lives inside a role's turn (e.g. Kimi's
        ``<|im_middle|>`` between role name and body) and so must not
        split the segment.
    assistant_start_str : str
        Decoded text prefix identifying assistant content segments, e.g.
        ``"<|im_start|>assistant"`` (Qwen3) or ``"<|assistant|>"`` (GLM).
        Used to classify content mismatches as assistant vs non-assistant.
    trim_trailing_ids : set[int] | None
        Token IDs to strip from both sequence tails before comparison
        (see :func:`_trim_trailing`).  Stored as a default; callers of
        :meth:`compare_sequences` may supply additional IDs that are
        **unioned** with this set.
    """

    def __init__(
        self,
        tokenizer,
        assistant_start_str: str,
        special_token_ids: set[int] | None = None,
        trim_trailing_ids: set[int] | None = None,
    ):
        # [FUNCTION DOC] Stores the tokenizer and the assistant start string. Special ids: a copy
        # of the given set, or, if none was given, the auto-detected set from the tokenizer.
        # Trailing ids: a copy of the given set, or none if the set is missing or empty.
        self.tokenizer = tokenizer
        if special_token_ids is not None:
            self._special_ids = set(special_token_ids)
        else:
            self._special_ids = self.collect_special_ids(tokenizer)
        self._assistant_start_str = assistant_start_str
        self._trim_trailing_ids: set[int] | None = set(trim_trailing_ids) if trim_trailing_ids else None

    @staticmethod
    def collect_special_ids(tokenizer) -> set[int]:
        # [FUNCTION DOC] Static method. Returns the union of the tokenizer's all-special ids and,
        # when the tokenizer has an added-tokens mapping, every added token id whose token is
        # flagged special. Reads only; no side effects.
        """Collect token IDs with ``special=True`` from the tokenizer.

        Special tokens are structural markers added by the chat template to
        delimit messages, roles, and control flow — for example
        ``<|im_start|>``, ``<|im_end|>``, ``<|endoftext|>``, ``<s>``,
        ``</s>``, and ``<|assistant|>``.

        Tokens that encode *content* produced by a role are **not** special,
        even if they look "special" to a human.  For instance, ``<think>`` and
        ``</think>`` in reasoning models are regular content tokens generated
        by the assistant — the tokenizer does not flag them as
        ``special=True``, so they are not collected here.
        """
        ids = set(tokenizer.all_special_ids)
        decoder = getattr(tokenizer, "added_tokens_decoder", None)
        if decoder:
            ids |= {k for k, v in decoder.items() if v.special}
        return ids

    def segment_by_special_tokens(self, token_ids: list[int]) -> list[Segment]:
        # [FUNCTION DOC] Single left-to-right pass. Empty input gives an empty list. Each token in
        # the special set becomes its own special segment (flushing any pending content run first);
        # other tokens accumulate into a content segment, flushed at the next special token and at
        # the end.
        """Split *token_ids* into segments at special-token boundaries.

        Each special token becomes its own single-ID segment with
        ``is_special=True``.  Consecutive non-special tokens are grouped
        into content segments.  Example for Qwen3::

            [<|im_start|>, "assistant", "\\n", "Hi", <|im_end|>, "\\n"]
            → [special(<|im_start|>), content("assistant\\nHi"), special(<|im_end|>), content("\\n")]
        """
        if not token_ids:
            return []

        segments: list[Segment] = []
        current: list[int] = []
        for tid in token_ids:
            if tid in self._special_ids:
                if current:
                    segments.append(Segment(token_ids=current))
                    current = []
                segments.append(Segment(token_ids=[tid], is_special=True))
            else:
                current.append(tid)
        if current:
            segments.append(Segment(token_ids=current))
        return segments

    def compare_sequences(
        self,
        expected_ids: list[int],
        actual_ids: list[int],
        trim_trailing_ids: set[int] | None = None,
    ) -> list[Mismatch]:
        # [FUNCTION DOC] Steps: (1) trailing ids to trim are the constructor set unioned with the
        # extra argument; if non-empty, strip those ids from the tails of both sequences. (2)
        # Segment both. (3) If the structure check reports a difference, return it alone (single
        # element list). (4) Otherwise walk aligned segment pairs; a content pair counts as
        # assistant content only if both sides' segments are classified as assistant content; collect
        # every per-segment mismatch in order. Returns an empty list when the sequences agree.
        """Compare two token-ID sequences and return mismatches.

        Parameters
        ----------
        trim_trailing_ids : set[int] | None
            Additional token IDs to strip from both sequence tails before
            comparison.  **Unioned** with the IDs passed at construction time.
        """
        trim = self._trim_trailing_ids or set()
        if trim_trailing_ids:
            trim = trim | trim_trailing_ids
        if trim:
            expected_ids = _trim_trailing(expected_ids, trim)
            actual_ids = _trim_trailing(actual_ids, trim)

        exp_segs = self.segment_by_special_tokens(expected_ids)
        act_segs = self.segment_by_special_tokens(actual_ids)

        structural = self._check_segment_structure(exp_segs, act_segs)
        if structural:
            return [structural]

        mismatches: list[Mismatch] = []
        for idx, (exp, act) in enumerate(zip(exp_segs, act_segs, strict=True)):
            is_assistant_content = self._is_assistant_content(exp_segs, idx) and self._is_assistant_content(
                act_segs, idx
            )
            m = self._compare_single_segment(idx, exp, act, is_assistant_content=is_assistant_content)
            if m is not None:
                mismatches.append(m)
        return mismatches

    def _check_segment_structure(
        self,
        exp_segs: list[Segment],
        act_segs: list[Segment],
    ) -> Mismatch | None:
        # [FUNCTION DOC] HELPER: Returns nothing when both lists have the same length and the same
        # per-position special flags. Otherwise returns a special-token-count mismatch with segment
        # index -1, structure descriptions of both sides as the texts, and a detail saying whether
        # the count or the pattern differs (count is checked first).
        """Pre-check that expected and actual segment lists have the same count
        and the same special/content pattern before per-segment comparison."""
        if len(exp_segs) != len(act_segs):
            detail = f"segment count differs: expected {len(exp_segs)}, got {len(act_segs)}"
        elif [s.is_special for s in exp_segs] != [s.is_special for s in act_segs]:
            detail = "segment structure (special/content pattern) differs"
        else:
            return None
        return Mismatch(
            type=MismatchType.SPECIAL_TOKEN_COUNT,
            segment_index=-1,
            expected_text=self._describe_structure(exp_segs),
            actual_text=self._describe_structure(act_segs),
            detail=detail,
        )

    def _compare_single_segment(
        self,
        idx: int,
        exp: Segment,
        act: Segment,
        *,
        is_assistant_content: bool,
    ) -> Mismatch | None:
        # [FUNCTION DOC] HELPER: Special pair: returns a special-token-type mismatch when the id
        # lists differ, otherwise nothing. Content pair: decodes both (special tokens kept) and
        # compares exact text; equal gives nothing, different gives an assistant-text or
        # non-assistant-text mismatch depending on the keyword flag.
        # KNOWN ISSUE: the docstring says content is compared after stripping whitespace, but the
        # code compares the decoded text exactly (the strip was removed, per the inline comment).
        """Compare a single aligned segment pair and return a mismatch if they differ.

        Special segments are compared by token ID.  Content segments are decoded
        and compared as stripped text — leading/trailing whitespace (``\\n``,
        spaces) is ignored because chat templates may insert boundary newlines
        that differ from the TITO prefix.  This whitespace-only difference does
        not cause meaningful misalignment with the chat template, so we strip
        to avoid noisy false positives.
        """
        if exp.is_special:
            if exp.token_ids != act.token_ids:
                return Mismatch(
                    type=MismatchType.SPECIAL_TOKEN_TYPE,
                    segment_index=idx,
                    expected_text=self._decode(exp.token_ids),
                    actual_text=self._decode(act.token_ids),
                )
            return None

        # After ignoring assistant text diff, there is no need to keep the strip operator,
        # as other text should be exact match.
        exp_text = self._decode(exp.token_ids)
        act_text = self._decode(act.token_ids)
        if exp_text == act_text:
            return None

        return Mismatch(
            type=MismatchType.ASSISTANT_TEXT if is_assistant_content else MismatchType.NON_ASSISTANT_TEXT,
            segment_index=idx,
            expected_text=exp_text,
            actual_text=act_text,
        )

    def _is_assistant_content(self, segments: list[Segment], idx: int) -> bool:
        # [FUNCTION DOC] HELPER: Returns false when no assistant start string is configured, when
        # the segment is special, when it is the first segment, or when the previous segment is not
        # special. Otherwise decodes the previous special token and the first 20 tokens of this
        # segment separately, concatenates the texts, and returns whether the result starts with
        # the assistant start string.
        """Check if the content segment at *idx* belongs to an assistant message.

        Decodes the preceding special-token segment and the first few tokens of
        the current segment *separately*, then concatenates the decoded strings.
        If the result starts with ``assistant_start_str`` (e.g.
        ``"<|im_start|>assistant"``), this segment is classified as assistant
        content — mismatches there are expected and non-severe.

        """
        if self._assistant_start_str is None:
            return False
        if segments[idx].is_special:
            return False
        if idx == 0:
            return False
        prev = segments[idx - 1]
        if not prev.is_special:
            return False
        special_text = self._decode(prev.token_ids)
        # Decode enough prefix tokens to capture the role label (e.g. "assistant\n").
        content_prefix = self._decode(segments[idx].token_ids[:20])
        return (special_text + content_prefix).startswith(self._assistant_start_str)

    def _decode(self, token_ids: list[int]) -> str:
        # [FUNCTION DOC] HELPER: Decodes token ids to text keeping special tokens visible.
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def _describe_structure(self, segments: list[Segment]) -> str:
        # [FUNCTION DOC] HELPER: Returns a space-separated summary: each special segment as its
        # decoded text in square brackets, each content segment as its token count in parentheses.
        return " ".join(
            f"[{self._decode(s.token_ids)}]" if s.is_special else f"({len(s.token_ids)} tokens)" for s in segments
        )


def _trim_trailing(ids: list[int], to_remove: set[int]) -> list[int]:
    # [FUNCTION DOC] HELPER: Returns a new list with the longest tail made only of ids from the
    # removal set cut off; may return an empty list. The input is not modified.
    """Strip trailing token IDs that belong to *to_remove*.

    The model's generated output typically ends with a stop token (e.g.
    ``<|observation|>`` for GLM, ``<|im_end|>`` for Qwen) that won't appear
    at the same position in the template-rendered expected sequence.  Stripping
    these trailing tokens from both sides before comparison avoids false
    structural mismatches.
    """
    end = len(ids)
    while end > 0 and ids[end - 1] in to_remove:
        end -= 1
    return ids[:end]
