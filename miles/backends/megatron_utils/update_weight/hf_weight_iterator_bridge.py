# [MODULE DOC] Weight iterator implementation that uses Megatron Bridge's AutoBridge to export
# base weights and LoRA adapter weights as Hugging Face-named tensors, then applies the repo's own
# post-processing (padding and renaming) and optional quantization. Also holds a helper that swaps
# the bridge's conversion tasks' weights for the live (possibly offloaded or backed-up) weights,
# and a small sized lazy-mapping wrapper used for that.
import dataclasses
import inspect
import itertools

from miles.backends.megatron_utils.update_weight.hf_weight_iterator import (
    MegatronHfWeightIteratorBase,
    _iter_mm_tower_units,
)
from miles.utils import megatron_bridge_utils
from miles.utils.lora.utils import is_lora_weight_name

from ..megatron_to_hf import postprocess_hf_param
from ..megatron_to_hf.processors import quantize_params
from ..misc_utils import strip_param_name_prefix


class HfWeightIteratorBridge(MegatronHfWeightIteratorBase):
    # [CLASS DOC] Iterator using a Megatron Bridge object (created once at construction from the
    # Hugging Face checkpoint, trusting remote code) for all name and layout conversion. Inherits
    # the pipeline-gather placement from its base. Base weights yield units of one or more tensors
    # (a quantized weight together with its scales); LoRA adapter tensors are exported separately.

    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC] Runs the base constructor, then imports Megatron Bridge lazily and stores
        # an AutoBridge built from the checkpoint path in arguments (reads checkpoint config,
        # may involve disk or network I/O).
        super().__init__(*args, **kwargs)

        from megatron.bridge import AutoBridge

        self._bridge = AutoBridge.from_hf_pretrained(self.args.hf_checkpoint, trust_remote_code=True)

    def _iter_hf_param_units(self, weights, *, materialize):
        # [FUNCTION DOC] Generator of units of Hugging Face tensors for base weights. Input: a
        # mapping from Megatron-local name (with wrapper prefix) to tensor, and a materialize flag.
        # Steps: strips the wrapper prefix from the mapping keys; inside the bridge's model-patching
        # context, gets the bridge's conversion tasks for the model, replaces their weights with the
        # given ones (lazily, moving them to the GPU), and starts the bridge export onto the GPU
        # without merging adapter weights, requesting the source Megatron names when the bridge
        # supports it. When not materializing, drains the export (so the internal tensor-parallel
        # collectives run on every rank) and stops without yielding. Otherwise post-processes and
        # quantizes as base weights, groups consecutive items by their source-name tuple, drops
        # LoRA-named tensors, and yields each nonempty group as one unit so a quantized weight and
        # its scales stay together. After the bridge context, yields the multimodal tower units
        # (reached only on the materializing path, because the non-materializing path returns early).
        renamed_megatron_local_weights = {strip_param_name_prefix(k): v for k, v in weights.items()}
        with megatron_bridge_utils.patch_megatron_model(self.model):
            conversion_tasks = self._bridge.get_conversion_tasks(self.model)
            conversion_tasks = _process_conversion_tasks(conversion_tasks, renamed_megatron_local_weights)
            named_weights = self._bridge.export_hf_weights(
                self.model,
                cpu=False,
                conversion_tasks=conversion_tasks,
                merge_adapter_weights=False,
                **self._source_name_kwargs(self._bridge.export_hf_weights),
            )

            # Apply postprocess + quantization (when targeting a quantized rollout,
            # e.g. FP8 sglang): base weights are quantized to match the rollout's
            # storage format so update_weights_from_tensor lands real weight + scale
            # pairs.
            if not materialize:
                # The export's internal TP collectives must still run on every rank.
                for _ in named_weights:
                    pass
                return

            named_weights = self._postprocess_and_quantize(named_weights, "base")
            # Group by the (tuple of) source names so quantize's weight + scales land in one unit.
            for _megatron_name, group in itertools.groupby(named_weights, key=lambda item: item[2]):
                unit = [(h, w) for h, w, _m in group if not is_lora_weight_name(h)]
                if unit:
                    yield unit
        yield from _iter_mm_tower_units(self.args, materialize=materialize)

    def _export_pp_local_lora(self, adapter):
        # [FUNCTION DOC] Implements the base hook. With no adapter object, exports the currently
        # active adapter. Otherwise (multi-LoRA) imports the slot-exposing helper lazily, exposes
        # the adapter's slot on the model while exporting, then slices each exported tensor down to
        # the adapter's rank via the slot helper. Returns a list of name and tensor pairs.
        if adapter is None:
            return self._export_current_adapter()

        from megatron.bridge.peft.multi_lora_layers import expose_adapter_slot

        from ..lora.slots import slice_lora_to_rank

        with expose_adapter_slot(self.model, adapter.slot):
            named_tensors = self._export_current_adapter()
        return [(h, slice_lora_to_rank(h, w, adapter.rank)) for h, w in named_tensors]

    def _export_current_adapter(self) -> list:
        # [FUNCTION DOC] HELPER: Inside the bridge's model-patching context, exports adapter weights
        # on the GPU with no progress display (requesting source names when supported), runs the
        # post-processing/quantization pipeline marked as adapter weights, and returns the list of
        # name and tensor pairs whose names look like LoRA weights.
        with megatron_bridge_utils.patch_megatron_model(self.model):
            named_weights = self._bridge.export_adapter_weights(
                self.model,
                cpu=False,
                show_progress=False,
                **self._source_name_kwargs(self._bridge.export_adapter_weights),
            )
            named_weights = self._postprocess_and_quantize(named_weights, "lora")
            return [(h, w) for h, w, _m in named_weights if is_lora_weight_name(h)]

    @staticmethod
    def _source_name_kwargs(export_fn) -> dict:
        # [FUNCTION DOC] HELPER: Static. Inspects the given export function's parameters; returns a
        # mapping asking for Megatron source names to be included when it has such a parameter,
        # else an empty mapping (older bridge versions).
        """Request source Megatron names via ``with_megatron_names`` when the bridge accepts it (main only)."""
        if "with_megatron_names" in inspect.signature(export_fn).parameters:
            return {"with_megatron_names": True}
        return {}

    @staticmethod
    def _source_names(item) -> tuple:
        # [FUNCTION DOC] HELPER: Static. Returns the source Megatron names carried as the third item
        # of an export item as a tuple: empty if the item has fewer than three fields or the field
        # is none, a one-element tuple for a single string, otherwise the field converted to a tuple.
        """Normalize the third field to a tuple of 0/1/N source Megatron names (a single str on the bridge branch)."""
        source = item[2] if len(item) > 2 else None
        if source is None:
            return ()
        if isinstance(source, str):
            return (source,)
        return tuple(source)

    def _postprocess_and_quantize(self, named_weights, weight_type: str):
        # [FUNCTION DOC] HELPER: Generator over export items (name, tensor, optional source names)
        # yielding triples (name, tensor, source-name tuple). Per item: removes the adapter
        # base-layer segment from the Hugging Face name; applies the post-processing step with the
        # first source Megatron name (none if there are none). If the weight type is "base", a
        # quantization config exists and a source name exists, builds the wrapper-prefixed Megatron
        # name and yields each tensor returned by the quantizer (weight and scales, with the packed
        # weight names) carrying the original source names; otherwise passes through unchanged.
        # Tensors without a Megatron source are treated as passthrough and never quantized.
        for item in named_weights:
            hf_param_name, weight = item[0], item[1]
            megatron_param_names = self._source_names(item)
            # Padding/quantization rules key on a Megatron name; packed grouped-expert tensors use the first source.
            megatron_param_name = megatron_param_names[0] if megatron_param_names else None
            hf_name = hf_param_name.replace(".base_layer.", ".")
            weight = postprocess_hf_param(
                args=self.args,
                megatron_param_name=megatron_param_name,
                hf_param_name=hf_name,
                param=weight,
            )
            if weight_type == "base" and self.quantization_config is not None and megatron_param_name is not None:
                # quantize_params expects the megatron name with the `module.module.`
                # prefix that the direct iterator uses; the bridge yields it without.
                # A tensor with no Megatron source (HF-only passthrough) is not a trainable weight: pass it through.
                qmegatron_name = f"module.module.{megatron_param_name}"
                for q_hf_name, q_weight in quantize_params(
                    self.args,
                    qmegatron_name,
                    [(hf_name, weight)],
                    self.quantization_config,
                    self.packed_weight_basenames,
                ):
                    yield q_hf_name, q_weight, megatron_param_names
            else:
                yield hf_name, weight, megatron_param_names


def _process_conversion_tasks(vanilla_conversion_tasks, new_weight_dict):
    # [FUNCTION DOC] HELPER: Wraps the bridge's conversion tasks so that each task, when iterated,
    # carries the weight from the given mapping instead of the model's own. Returns a sized lazy
    # sequence (length of the input); nothing is processed until iteration, and each iteration
    # re-applies the function.
    def _handle_one(task):
        # [FUNCTION DOC] HELPER: Returns the task unchanged if it is none (no Hugging Face mapping),
        # has no parameter weight, or its key (virtual stage plus parameter name) is not in the
        # mapping (buffer-like parameters). Otherwise returns a copy of the task whose parameter
        # weight is the mapped tensor moved to the current GPU.
        if task is None:
            # no HF mapping (e.g. Gemma-4 post_shared_expert_layernorm)
            return task
        if task.param_weight is None:
            return task

        weight_dict_key = f"vp_stages.{task.vp_stage}.{task.param_name}"
        if weight_dict_key not in new_weight_dict:
            # buffer-like params (Gemma-4 layer_scalar/scale) aren't in optimizer state; keep as-is
            return task
        new_param_weight = new_weight_dict[weight_dict_key]
        new_param_weight = new_param_weight.cuda()
        return dataclasses.replace(task, param_weight=new_param_weight)

    return _MapWithLen(_handle_one, vanilla_conversion_tasks)


class _MapWithLen:
    # [CLASS DOC] HELPER: Re-iterable lazy map that, unlike a plain map object, supports length
    # (taken from the underlying sequence) and can be iterated repeatedly; each iteration applies
    # the function afresh to each element.
    def __init__(self, fn, xs):
        # [FUNCTION DOC] HELPER: Stores the function and the underlying sized iterable.
        self.fn = fn
        self.xs = xs

    def __len__(self):
        # [FUNCTION DOC] HELPER: Returns the length of the underlying iterable.
        return len(self.xs)

    def __iter__(self):
        # [FUNCTION DOC] HELPER: Generator yielding the function applied to each element in order.
        for x in self.xs:
            yield self.fn(x)
