# [MODULE DOC] Core data types shared along the rollout-to-training path: the weight-version span
# records attached to generated tokens, a per-sample reward spec, the Sample dataclass (prompt,
# generated tokens, rewards, rollout-side log probabilities and routing records, status, metadata)
# with serialization, validation, token stripping, retry reset and update-from-engine-metadata
# behavior, a parameter description record, the rollout batch alias, and multimodal type constants.
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any

import numpy
import torch

from miles.utils.sampling_mask import RolloutSamplingMask

LEGACY_WEIGHT_VERSIONS_KEY = "legacy_weight_versions"


@dataclass(frozen=True)
class WeightVersionSpan:
    # [CLASS DOC] Immutable record that tokens in the half-open absolute range from start to end
    # (positions in the sample's full token list, prompt included) were generated under the given
    # weight version string.
    version: str
    abs_start: int
    abs_end: int


@dataclass
class WeightVersionsPerCall:
    # [CLASS DOC] Weight version spans produced by one generation call (mutable list of spans).
    spans: list[WeightVersionSpan] = field(default_factory=list)

    def to_dicts(self) -> list[dict]:
        # [FUNCTION DOC] Returns each span as a mapping of its three fields.
        return [asdict(span) for span in self.spans]

    @staticmethod
    def from_dicts(data: list[dict]) -> "WeightVersionsPerCall":
        # [FUNCTION DOC] Builds an object from a list of span mappings (unknown keys raise a type error).
        return WeightVersionsPerCall(spans=[WeightVersionSpan(**span) for span in data])

    @staticmethod
    def from_meta_info(meta_info: dict, output_end: int) -> "WeightVersionsPerCall":
        # [FUNCTION DOC] Converts the inference engine's per-call metadata into absolute spans.
        # The number of output tokens is the length of the output token logprobs list (missing means
        # zero). The call's output tokens occupy absolute positions from output_end minus that count to
        # output_end. If the metadata has a list of weight version spans (version, start, end relative
        # to the output), asserts every end is within the output token count; else if it has a single
        # weight version, asserts there is output (or no completion tokens) and uses one span covering
        # all output; otherwise no spans. Spans are shifted by the output start and empty spans
        # (start not below end) are dropped. Assertion errors explain that logprobs must be returned.
        num_output_tokens = len(meta_info.get("output_token_logprobs") or [])
        if (raw_spans := meta_info.get("weight_versions")) is not None:
            output_relative = [(span["version"], span["start"], span["end"]) for span in raw_spans]
            assert all(end <= num_output_tokens for _, _, end in output_relative), (
                f"weight version spans {raw_spans} extend past the {num_output_tokens} output tokens implied by "
                f"output_token_logprobs; anchoring spans relies on return_logprob reporting every output token"
            )
        elif meta_info.get("weight_version") is not None:
            assert num_output_tokens > 0 or not meta_info.get("completion_tokens"), (
                "weight_version is present but output_token_logprobs is empty; anchoring the version to output "
                "tokens requires return_logprob=True on the generate request"
            )
            output_relative = [(meta_info["weight_version"], 0, num_output_tokens)]
        else:
            output_relative = []

        output_start = output_end - num_output_tokens
        return WeightVersionsPerCall(
            spans=[
                WeightVersionSpan(version=version, abs_start=output_start + start, abs_end=output_start + end)
                for version, start, end in output_relative
                if start < end
            ]
        )


@dataclass(frozen=True)
class RewardSpec:
    # [CLASS DOC] Immutable per-sample reward dispatch override: optional reward model type and optional
    # custom reward function path (both default to none, meaning use the run-level setting).
    """Per-sample spec of how the response is scored; intentionally decoupled from adapter routing."""

    rm_type: str | None = None
    custom_rm_path: str | None = None


@dataclass
class Sample:
    # [CLASS DOC] One prompt/response sample flowing through rollout, reward and training. Field groups:
    # identity (group index, index, rollout id falling back to index); prompt (text or message list,
    # token ids covering prompt plus response, raw and processed multimodal inputs); response (text,
    # length in tokens, label, reward as number or mapping, per-token loss mask); weight versions per
    # generation call; rollout-side per-token records (log probabilities, top-k candidate ids and log
    # probs as arrays with rows per response token, sampling mask, routed experts and indexer top-k
    # arrays with one row per token except the last); teacher log probs and reverse-KL estimates for
    # on-policy distillation; a removal flag; a status (pending, completed, truncated, aborted,
    # failed); free-form metadata; generate function path; training metadata; reward spec override;
    # routing key; trainer model id; non-generation time; speculative decoding and prefix cache
    # statistics. Mutable.
    # KNOWN ISSUE: the multimodal input fields are annotated as mappings but default to none.
    """The sample generated"""

    group_index: int | None = None
    index: int | None = None
    # Rollout execution id; None falls back to ``index``. Compact / subagent
    # siblings must share it so the rollout is counted once.
    rollout_id: int | None = None
    # prompt
    prompt: str | list[dict[str, str]] = ""
    tokens: list[int] = field(default_factory=list)
    multimodal_inputs: dict[str, Any] = None  # raw multimodal data, e.g. images, videos, etc.
    multimodal_train_inputs: dict[str, Any] = None  # processed multimodal data, e.g. pixel_values, etc.
    # response
    response: str = ""
    response_length: int = 0
    label: str | None = None
    reward: float | dict[str, Any] | None = None
    loss_mask: list[int] | None = None
    weight_versions: list[WeightVersionsPerCall] = field(default_factory=list)
    rollout_log_probs: list[float] | None = None  # Log probabilities from rollout engine
    rollout_topk_token_ids: numpy.ndarray | None = None  # [response_length, k], int32; -1 pads unused slots
    rollout_topk_log_probs: numpy.ndarray | None = None  # [response_length, k], float32; -inf pads unused slots
    rollout_sampling_mask: RolloutSamplingMask | None = None
    rollout_routed_experts: numpy.ndarray | None = (
        None  # Routed experts from rollout engine. shape: (num_tokens-1, num_layers, moe_router_topk), dtype=int32
    )
    rollout_indexer_topk: numpy.ndarray | None = (
        None  # Indexer topk from rollout engine. shape: (num_tokens-1, num_indexer_layers, index_topk), dtype=int32
    )
    remove_sample: bool = False
    teacher_log_probs: list[float] | None = None  # Log probabilities from teacher model for OPD
    opd_reverse_kl: list[float] | None = None  # Precomputed per-token OPD reverse-KL estimate

    class Status(Enum):
        # [CLASS DOC] Generation outcome: pending (not generated), completed (stopped normally),
        # truncated (hit length limit), aborted, failed (recoverable failure, may hold partial output).
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"
        # Indicates a recoverable or non-critical failure during generation (e.g., tool call failure,
        # external API error, parsing error). Unlike ABORTED, FAILED samples may still contain partial
        # valid output and can be retried or handled gracefully.
        FAILED = "failed"

    status: Status = Status.PENDING

    metadata: dict = field(default_factory=dict)
    generate_function_path: str | None = None
    # metadata used during training, e.g., what loss to use for this sample.
    train_metadata: dict | None = None

    # Per-sample reward dispatch override
    reward_spec: RewardSpec | None = None

    # Per-sample routing key for the router's consistent_hashing policy (sent as X-SMG-Routing-Key)
    routing_key: str | None = None

    # Which policy model this sample trains and generates on; None when the run trains one policy
    trainer_model_id: str | None = None

    non_generation_time: float = 0.0  # time spent in non-generation steps

    @dataclass
    class SpecInfo:
        # [CLASS DOC] Accumulated speculative decoding counters: correct drafts, proposed drafts,
        # verification count, completion tokens.
        spec_num_correct_drafts: int = 0
        spec_num_proposed_drafts: int = 0
        spec_verify_ct: int = 0
        completion_tokens: int = 0

        @property
        def spec_accept_rate(self) -> float:
            # [FUNCTION DOC] Correct drafts divided by proposed drafts; zero when none proposed.
            if self.spec_num_proposed_drafts == 0:
                return 0.0
            return self.spec_num_correct_drafts / self.spec_num_proposed_drafts

        @property
        def spec_accept_length(self) -> float:
            # [FUNCTION DOC] Completion tokens divided by verification count; zero when no verification.
            return self.completion_tokens / self.spec_verify_ct if self.spec_verify_ct > 0 else 0.0

        def add(self, meta_info: dict):
            # [FUNCTION DOC] Adds one call's engine metadata: ignored entirely unless its verify count
            # is positive (missing or none counts as zero); otherwise adds correct drafts, proposed
            # drafts (missing as zero), the verify count and completion tokens (missing as zero).
            spec_verify_ct = meta_info.get("spec_verify_ct") or 0
            if spec_verify_ct <= 0:
                return
            self.spec_num_correct_drafts += meta_info.get("spec_num_correct_drafts", 0)
            self.spec_num_proposed_drafts += meta_info.get("spec_num_proposed_drafts", 0)
            self.spec_verify_ct += spec_verify_ct
            self.completion_tokens += meta_info.get("completion_tokens", 0)

        def to_dict(self):
            # [FUNCTION DOC] Returns the four counters as a mapping.
            return {
                "spec_num_correct_drafts": self.spec_num_correct_drafts,
                "spec_num_proposed_drafts": self.spec_num_proposed_drafts,
                "spec_verify_ct": self.spec_verify_ct,
                "completion_tokens": self.completion_tokens,
            }

        @staticmethod
        def from_dict(data: dict):
            # [FUNCTION DOC] Builds counters from a mapping, accepting legacy key names for correct
            # drafts (accepted token number), proposed drafts (draft token number) and completion
            # tokens (completion token number); missing values are zero.
            info = Sample.SpecInfo()
            info.spec_num_correct_drafts = data.get("spec_num_correct_drafts", data.get("spec_accept_token_num", 0))
            info.spec_num_proposed_drafts = data.get("spec_num_proposed_drafts", data.get("spec_draft_token_num", 0))
            info.spec_verify_ct = data.get("spec_verify_ct", 0)
            info.completion_tokens = data.get("completion_tokens", data.get("completion_token_num", 0))
            return info

    spec_info: SpecInfo = field(default_factory=SpecInfo)

    @dataclass
    class PrefixCacheInfo:
        # [CLASS DOC] Accumulated prefix cache counters: cached prompt tokens and total prompt tokens.
        cached_tokens: int = 0
        total_prompt_tokens: int = 0

        @property
        def prefix_cache_hit_rate(self) -> float:
            # [FUNCTION DOC] Cached tokens divided by total prompt tokens; zero when no prompt tokens.
            return self.cached_tokens / self.total_prompt_tokens if self.total_prompt_tokens > 0 else 0.0

        def add(self, meta_info: dict):
            # [FUNCTION DOC] Adds the call's cached token count and prompt token count (missing as
            # zero).
            self.cached_tokens += meta_info.get("cached_tokens", 0)
            # new_tokens = input_tokens - cached_tokens
            self.total_prompt_tokens += meta_info.get("prompt_tokens", 0)

        def to_dict(self):
            # [FUNCTION DOC] Returns both counters as a mapping.
            return {
                "cached_tokens": self.cached_tokens,
                "total_prompt_tokens": self.total_prompt_tokens,
            }

        @staticmethod
        def from_dict(data: dict):
            # [FUNCTION DOC] Builds counters from a mapping; missing values are zero.
            info = Sample.PrefixCacheInfo()
            info.cached_tokens = data.get("cached_tokens", 0)
            info.total_prompt_tokens = data.get("total_prompt_tokens", 0)
            return info

    prefix_cache_info: PrefixCacheInfo = field(default_factory=PrefixCacheInfo)

    def to_dict(self):
        # [FUNCTION DOC] Returns a shallow copy of the instance attribute mapping with status replaced by
        # its string value, speculative and prefix cache info replaced by their mappings, and weight
        # versions replaced by lists of span mappings per call. Other values (arrays, sampling mask,
        # metadata) are shared, not copied.
        value = self.__dict__.copy()
        value["status"] = self.status.value
        value["spec_info"] = self.spec_info.to_dict()
        value["prefix_cache_info"] = self.prefix_cache_info.to_dict()
        value["weight_versions"] = [call.to_dicts() for call in self.weight_versions]
        return value

    @staticmethod
    def from_dict(data: dict):
        # [FUNCTION DOC] Inverse of the dict conversion on a copy of the input. Status string is
        # converted to the enumeration (key error if missing, value error if unknown); speculative and
        # prefix cache infos are rebuilt (missing as empty). Weight versions: if the first entry is a
        # string (legacy format of a flat list of version strings) the strings are stored under the
        # legacy key as an extra attribute and weight versions are emptied; otherwise each entry is
        # rebuilt from span mappings. Known dataclass fields are passed to the constructor; keys that
        # are not fields are attached to the instance afterwards as plain attributes.
        data = dict(data)
        data["status"] = Sample.Status(data["status"])
        data["spec_info"] = Sample.SpecInfo.from_dict(data.get("spec_info", {}))
        data["prefix_cache_info"] = Sample.PrefixCacheInfo.from_dict(data.get("prefix_cache_info", {}))
        raw_weight_versions = data.get("weight_versions") or []

        if raw_weight_versions and isinstance(raw_weight_versions[0], str):
            data[LEGACY_WEIGHT_VERSIONS_KEY] = raw_weight_versions
            raw_weight_versions = []

        data["weight_versions"] = [WeightVersionsPerCall.from_dicts(call) for call in raw_weight_versions]

        field_names = set(Sample.__dataclass_fields__.keys())
        init_data = {k: v for k, v in data.items() if k in field_names}
        sample = Sample(**init_data)

        for key, value in data.items():
            if key not in field_names:
                setattr(sample, key, value)

        return sample

    def get_reward_value(self, args) -> float:
        # [FUNCTION DOC] Returns the reward itself when the reward-key argument is empty, otherwise the
        # reward mapping's entry for that key (errors if the reward is not a mapping or lacks the key).
        return self.reward if not args.reward_key else self.reward[args.reward_key]

    @property
    def effective_response_length(self):
        # [FUNCTION DOC] Number of response tokens that count for the loss: the sum of the loss mask if
        # present, else the response length.
        return sum(self.loss_mask) if self.loss_mask is not None else self.response_length

    def validate(self):
        # [FUNCTION DOC] Raises assertion errors (with messages) unless: response length is non-negative
        # and at most the token count; the loss mask, rollout log probs, teacher log probs and reverse-KL
        # lists (when present) each have exactly response-length entries; top-k ids and log probs are
        # both present or both absent, two-dimensional with identical shapes and one row per response
        # token; the sampling mask length equals the response length; routed experts length equals token
        # count minus one, or that plus extra positions contributed by media expansion (sum over the
        # vision-patch and audio-token counts in the multimodal train inputs of each count minus one);
        # indexer top-k length equals token count minus one; and the weight version spans, in order
        # across calls, each have a non-empty version, start at or after the previous span's end, be
        # non-empty and end within the token count.
        assert self.response_length >= 0, f"response_length must be >= 0, got {self.response_length}"
        assert (
            len(self.tokens) >= self.response_length
        ), f"tokens length ({len(self.tokens)}) must be >= response_length ({self.response_length})"
        if self.loss_mask is not None:
            assert (
                len(self.loss_mask) == self.response_length
            ), f"loss_mask length ({len(self.loss_mask)}) != response_length ({self.response_length})"
        if self.rollout_log_probs is not None:
            assert (
                len(self.rollout_log_probs) == self.response_length
            ), f"rollout_log_probs length ({len(self.rollout_log_probs)}) != response_length ({self.response_length})"
        if self.rollout_topk_token_ids is not None or self.rollout_topk_log_probs is not None:
            ids, logps = self.rollout_topk_token_ids, self.rollout_topk_log_probs
            assert (
                ids is not None and logps is not None
            ), "Rollout top-k logprobs IDs and probabilities must both be present"
            assert ids.ndim == 2 and ids.shape == logps.shape, "Rollout top-k logprobs candidate shapes do not match"
            assert (
                ids.shape[0] == self.response_length
            ), "Rollout top-k logprobs candidates do not match response length"
        if self.rollout_sampling_mask is not None:
            assert len(self.rollout_sampling_mask) == self.response_length, (
                f"rollout_sampling_mask length ({len(self.rollout_sampling_mask)}) "
                f"!= response_length ({self.response_length})"
            )
        if self.teacher_log_probs is not None:
            assert (
                len(self.teacher_log_probs) == self.response_length
            ), f"teacher_log_probs length ({len(self.teacher_log_probs)}) != response_length ({self.response_length})"
        if self.opd_reverse_kl is not None:
            assert (
                len(self.opd_reverse_kl) == self.response_length
            ), f"opd_reverse_kl length ({len(self.opd_reverse_kl)}) != response_length ({self.response_length})"
        if self.rollout_routed_experts is not None:
            actual = len(self.rollout_routed_experts)
            expect = len(self.tokens) - 1
            mm = self.multimodal_train_inputs or {}
            extra = sum(
                int(c) - 1 for key in ("mm_vision_num_patches", "mm_audio_num_tokens") for c in list(mm.get(key) or [])
            )
            assert actual in (expect, expect + extra), (
                f"rollout_routed_experts length ({actual}) != len(tokens) - 1 ({expect})"
                f" or media-expanded ({expect + extra})"
            )
        if self.rollout_indexer_topk is not None:
            actual = len(self.rollout_indexer_topk)
            expect = len(self.tokens) - 1
            assert actual == expect, f"rollout_indexer_topk length ({actual}) != len(tokens) - 1 ({expect})"
        previous_end = 0
        for span in self.all_weight_version_spans:
            assert span.version, f"weight version span has empty version: {self.weight_versions}"
            assert (
                previous_end <= span.abs_start < span.abs_end <= len(self.tokens)
            ), f"invalid weight version span {span} (previous_end={previous_end}, len(tokens)={len(self.tokens)})"
            previous_end = span.abs_end

    def strip_last_output_tokens(self, n: int, tokenizer) -> None:
        # [FUNCTION DOC] No-op for n not positive; asserts n does not exceed the response length.
        # Mutates in place: drops the last n tokens, reduces response length, trims the top-k arrays,
        # rollout log probs, teacher log probs, reverse-KL list, loss mask, routed experts and indexer
        # top-k arrays by n entries, replaces the sampling mask by its prefix, trims the optional
        # student top log probs list in the metadata, re-decodes the response text from the remaining
        # response tokens (empty string when none remain), and clips weight version spans: spans
        # starting at or beyond the new token count are removed and spans ending beyond it are cut.
        # The reward, status and cache statistics are untouched. Note slicing with a negative count on
        # routed experts and indexer arrays assumes they are aligned with the end of the tokens.
        """Remove the last *n* output tokens and all associated per-token info."""
        if n <= 0:
            return
        assert (
            n <= self.response_length
        ), f"cannot strip {n} tokens: only {self.response_length} output tokens available"
        self.tokens = self.tokens[:-n]
        self.response_length -= n
        if self.rollout_topk_token_ids is not None:
            self.rollout_topk_token_ids = self.rollout_topk_token_ids[:-n]
            self.rollout_topk_log_probs = self.rollout_topk_log_probs[:-n]
        if self.rollout_log_probs is not None:
            self.rollout_log_probs = self.rollout_log_probs[:-n]
        if self.rollout_sampling_mask is not None:
            self.rollout_sampling_mask = self.rollout_sampling_mask.prefix(self.response_length)
        if self.teacher_log_probs is not None:
            self.teacher_log_probs = self.teacher_log_probs[:-n]
        if self.opd_reverse_kl is not None:
            self.opd_reverse_kl = self.opd_reverse_kl[:-n]
        if self.metadata and "opd_student_top_logprobs" in self.metadata:
            self.metadata["opd_student_top_logprobs"] = self.metadata["opd_student_top_logprobs"][:-n]
        if self.loss_mask is not None:
            self.loss_mask = self.loss_mask[:-n]
        self.response = tokenizer.decode(self.tokens[-self.response_length :]) if self.response_length > 0 else ""
        if self.rollout_routed_experts is not None:
            self.rollout_routed_experts = self.rollout_routed_experts[:-n]
        if self.rollout_indexer_topk is not None:
            self.rollout_indexer_topk = self.rollout_indexer_topk[:-n]
        for call in self.weight_versions:
            call.spans = [
                span if span.abs_end <= len(self.tokens) else replace(span, abs_end=len(self.tokens))
                for span in call.spans
                if span.abs_start < len(self.tokens)
            ]

    def reset_for_retry(self) -> None:
        # [FUNCTION DOC] Mutates in place: clears tokens, processed multimodal inputs, response text and
        # length, reward, loss mask, weight versions, rollout log probs, top-k arrays, sampling mask,
        # routed experts and indexer top-k; sets status to aborted; clears trainer model id and
        # non-generation time; replaces speculative and prefix cache statistics with fresh ones; clears
        # the removal flag and training metadata.
        # KNOWN ISSUE: the docstring says everything else is restored to dataclass defaults, but status
        # is set to aborted (the default is pending), and the teacher log probs, reverse-KL list,
        # rollout id, reward spec and raw multimodal inputs are left unchanged.
        """Reset generated outputs so the original prompt can be re-sampled.

        Keeps identity / prompt fields (group_index, index, prompt, label,
        multimodal_inputs, metadata, generate_function_path, routing_key) and
        restores everything else to dataclass defaults.
        """
        self.tokens = []
        self.multimodal_train_inputs = None
        self.response = ""
        self.response_length = 0
        self.reward = None
        self.loss_mask = None
        self.weight_versions = []
        self.rollout_log_probs = None
        self.rollout_topk_token_ids = None
        self.rollout_topk_log_probs = None
        self.rollout_sampling_mask = None
        self.rollout_routed_experts = None
        self.rollout_indexer_topk = None
        self.status = Sample.Status.ABORTED
        self.trainer_model_id = None
        self.non_generation_time = 0.0
        self.spec_info = Sample.SpecInfo()
        self.prefix_cache_info = Sample.PrefixCacheInfo()
        self.remove_sample = False
        self.train_metadata = None

    @property
    def all_weight_version_spans(self) -> list[WeightVersionSpan]:
        # [FUNCTION DOC] All spans of all calls flattened in call order.
        return [span for call in self.weight_versions for span in call.spans]

    @property
    def oldest_weight_version(self) -> int | None:
        # [FUNCTION DOC] Minimum integer among span versions that consist only of decimal digits, or
        # none if there are none.
        """Minimum weight version across all turns (generation calls) for this trajectory."""
        numeric = [int(span.version) for span in self.all_weight_version_spans if str(span.version).isdigit()]
        return min(numeric) if numeric else None

    def update_from_meta_info(self, args, meta_info: dict):
        # [FUNCTION DOC] Folds one generation call's engine metadata into the sample (mutating it). If
        # a speculative algorithm is configured in the args, adds the metadata to the speculative
        # statistics; always adds to prefix cache statistics; appends the weight version spans for this
        # call anchored at the current token count (so tokens must already include this call's output);
        # then sets status from the finish reason type: length gives truncated, abort gives aborted,
        # stop gives completed; any other type leaves the status unchanged. Raises key errors if the
        # finish reason is missing.
        """
        Update the sample with new information from meta_info returned by the rollout engine.
        And extract
        """
        if args.sglang_speculative_algorithm:
            # cannot directly use spec info from sglang because of partial rollout.
            self.spec_info.add(meta_info=meta_info)

        # Collect prefix cache statistics
        self.prefix_cache_info.add(meta_info=meta_info)

        self.weight_versions.append(WeightVersionsPerCall.from_meta_info(meta_info, output_end=len(self.tokens)))

        match meta_info["finish_reason"]["type"]:
            case "length":
                self.status = Sample.Status.TRUNCATED
            case "abort":
                self.status = Sample.Status.ABORTED
            case "stop":
                self.status = Sample.Status.COMPLETED


@dataclass(frozen=True)
class ParamInfo:
    # [CLASS DOC] Immutable description of one model parameter: name, dtype, shape, attribute mapping,
    # size, and the rank that holds the source copy.
    name: str
    dtype: torch.dtype
    shape: torch.Size
    attrs: dict
    size: int
    src_rank: int


# A dict-based batch produced along the rollout -> training path
# In Megatron backend, several fields are converted to torch.Tensor lists on GPU
# before being consumed by data iterators (see megatron_utils.actor._get_rollout_data).
RolloutBatch = dict[str, list[torch.Tensor] | list[numpy.ndarray] | list[int] | list[float] | list[str]]


@dataclass
class MultimodalType:
    # [CLASS DOC] A media kind: the identifier used in message content and the placeholder token used in
    # conversation text.
    name: str  # Type identifier used in message content (e.g., "image")
    placeholder: str  # Placeholder token in conversation messages (e.g., "<image>")


class MultimodalTypes:
    # [CLASS DOC] Namespace of the three supported media kinds (image, video, audio) with their
    # placeholders, plus lookup helpers.
    IMAGE = MultimodalType(name="image", placeholder="<image>")
    VIDEO = MultimodalType(name="video", placeholder="<video>")
    AUDIO = MultimodalType(name="audio", placeholder="<audio>")

    @classmethod
    def all(cls) -> list[MultimodalType]:
        # [FUNCTION DOC] The three kinds in order image, video, audio.
        return [cls.IMAGE, cls.VIDEO, cls.AUDIO]

    @classmethod
    def get(cls, name: str) -> MultimodalType | None:
        # [FUNCTION DOC] The kind with the given name, or none if unknown.
        return next((m for m in cls.all() if m.name == name), None)
