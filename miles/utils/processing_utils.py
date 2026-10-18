# [MODULE DOC]
# Tokenizer and multimodal processor helpers: cached tokenizer loading (with optional custom chat
# template and a fix for transformers v5 tokenizer component mismatches), processor loading
# (special-cased for inkling), building processor call arguments, calling processors (with the
# media-list convention of kimi models), extracting trainable multimodal inputs, extracting vision
# inputs from prompts, and encoding images as data urls for the rollout engine.
import base64
import inspect
import io
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer as RawTokenizer
from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase, ProcessorMixin

from miles.utils.hf_utils.config import register_hf_config_aliases

logger = logging.getLogger(__name__)


def _fix_v5_tokenizer_components(tokenizer: PreTrainedTokenizerBase, model_name_or_path: str) -> None:
    # [FUNCTION DOC] HELPER: Repairs a loaded fast tokenizer in place. Does nothing if it has no
    # backend tokenizer. Reads the raw tokenizer definition file from the local model directory, or
    # from the local hub cache only (no network); any failure is logged as a warning and ends the
    # fix. If the raw pre-tokenizer type name differs from the loaded one (both present), logs and
    # replaces the loaded pre-tokenizer and decoder by the raw ones. Mutates the tokenizer.
    # transformers v5's LlamaTokenizerFast rebuilds pre_tokenizer/decoder in
    # __init__, discarding the originals from tokenizer.json.  DeepSeek-V3.2
    # declares LlamaTokenizerFast but actually uses ByteLevel, so without this
    # fix the loaded tokenizer decodes Metaspace ▁ instead of ByteLevel Ġ/Ċ
    # and diverges from the sglang-served tokenizer.  Mirrors sglang's
    # _fix_v5_tokenizer_components (hf_transformers_utils.py).
    backend = getattr(tokenizer, "_tokenizer", None)
    if backend is None:
        return

    try:
        local_path = Path(model_name_or_path) / "tokenizer.json"
        if local_path.is_file():
            tok_file = str(local_path)
        else:
            tok_file = hf_hub_download(model_name_or_path, "tokenizer.json", local_files_only=True)
        raw = RawTokenizer.from_file(tok_file)
    except Exception as e:
        logger.warning("Could not load tokenizer.json for %s: %s", model_name_or_path, e)
        return

    raw_pre = type(raw.pre_tokenizer).__name__ if raw.pre_tokenizer else None
    loaded_pre = type(backend.pre_tokenizer).__name__ if backend.pre_tokenizer else None

    if raw_pre and loaded_pre and raw_pre != loaded_pre:
        logger.info(
            "Fixing v5 tokenizer component mismatch for %s: pre_tokenizer %s -> %s, decoder %s -> %s",
            model_name_or_path,
            loaded_pre,
            raw_pre,
            type(backend.decoder).__name__ if backend.decoder else None,
            type(raw.decoder).__name__ if raw.decoder else None,
        )
        backend.pre_tokenizer = raw.pre_tokenizer
        backend.decoder = raw.decoder


# Default image patch size for vision-language models
# Note: Qwen3-VL uses 16, Qwen2.5-VL uses 14
# Reference: https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/README.md
DEFAULT_PATCH_SIZE = 14


_TOKENIZER_CACHE: dict[tuple, PreTrainedTokenizerBase] = {}


def _make_cache_key(name_or_path: str, chat_template_path: str | None, kwargs: dict) -> tuple | None:
    # [FUNCTION DOC] HELPER: Cache key of name, template path and sorted keyword items; none if the
    # keyword values are unhashable (or unsortable) so such loads are not cached. Note: key sorting
    # and hashing failures other than a type error would propagate.
    try:
        kwargs_items = tuple(sorted(kwargs.items()))
        hash(kwargs_items)
    except TypeError:
        return None
    return (name_or_path, chat_template_path, kwargs_items)


def load_tokenizer(name_or_path: str, chat_template_path: str | None = None, **kwargs) -> PreTrainedTokenizerBase:
    # [FUNCTION DOC] Returns a cached tokenizer if the same arguments were loaded before (the same
    # object is shared, so mutating it affects all users). Otherwise registers model type aliases,
    # loads with the auto tokenizer (extra keyword arguments passed through), applies the v5
    # component fix, and if a chat template path is given asserts the file exists and replaces the
    # tokenizer's chat template with its text. Caches when the key is hashable.
    # Cache keyed by (name, chat_template_path, kwargs) — the fast suite creates
    # hundreds of SessionServer / MockSGLangServer fixtures and each previously
    # triggered a fresh AutoTokenizer.from_pretrained, tripping HF Hub rate limits.
    cache_key = _make_cache_key(name_or_path, chat_template_path, kwargs)
    if cache_key is not None and cache_key in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[cache_key]

    register_hf_config_aliases()
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    _fix_v5_tokenizer_components(tokenizer, name_or_path)
    if chat_template_path:
        assert os.path.isfile(chat_template_path), (
            f"chat_template_path not found: {chat_template_path}. "
            f"Ensure the path is accessible on this node (e.g. inside the miles repo or on a shared filesystem)."
        )
        with open(chat_template_path) as f:
            tokenizer.chat_template = f.read()
        logger.info("Loaded custom chat template from %s", chat_template_path)

    if cache_key is not None:
        _TOKENIZER_CACHE[cache_key] = tokenizer
    return tokenizer


def build_processor_kwargs(multimodal_inputs: dict | None = None) -> dict:
    # [FUNCTION DOC] Copies the multimodal inputs mapping (empty if none) and forces per-modality
    # keyword groups: text keyword group gets no tensor type (plain lists), and audio, images and
    # videos groups get tensor type "pt" (merged over any existing group contents; the forced value
    # wins). Returns the new mapping; the input is not mutated.

    modality_forced = {"return_tensors": "pt"}

    result = dict(multimodal_inputs) if multimodal_inputs else {}

    # return_tensors=None for text (input_ids), "pt" for modality-specific outputs.
    # Use per-modality dicts to avoid transformers >=5.0 duplicate kwarg error.
    result["text_kwargs"] = {**result.get("text_kwargs", {}), "return_tensors": None}
    for key in ("audio_kwargs", "images_kwargs", "videos_kwargs"):
        if key in result:
            result[key] = {**result[key], **modality_forced}
        else:
            result[key] = modality_forced.copy()

    return result


def processor_requires_medias(processor) -> bool:
    # [FUNCTION DOC] True if the processor's call signature has both a medias and a text parameter;
    # if the signature cannot be inspected, true when the processor has a media processor attribute.
    try:
        params = inspect.signature(processor).parameters
        return "medias" in params and "text" in params
    except (TypeError, ValueError):
        return hasattr(processor, "media_processor")


def call_processor(processor, text, multimodal_inputs: dict | None = None):
    # [FUNCTION DOC] Runs the processor on the text. For media-list processors, builds a list of
    # typed media entries (images then videos, from the lists under those keys) and calls with text
    # and medias (other multimodal keys are ignored). Otherwise calls with text and the keyword
    # groups from the kwargs builder. Returns the processor output.
    multimodal_inputs = multimodal_inputs or {}

    # for kimi-vl & kimi-2.5
    if processor_requires_medias(processor):
        medias = []
        if images := multimodal_inputs.get("images"):
            medias.extend({"type": "image", "image": image} for image in images)
        if videos := multimodal_inputs.get("videos"):
            medias.extend({"type": "video", "video": video} for video in videos)
        return processor(text=text, medias=medias)

    kwargs = build_processor_kwargs(multimodal_inputs)
    return processor(text=text, **kwargs)


def extract_multimodal_train_inputs(processor_output: Mapping[str, Any]) -> dict[str, Any] | None:
    # [FUNCTION DOC] Returns the processor output entries other than input ids, attention mask and
    # multimodal token type ids; none when nothing remains.
    excluded_keys = {"input_ids", "attention_mask", "mm_token_type_ids"}
    return {key: value for key, value in processor_output.items() if key not in excluded_keys} or None


def load_processor(name_or_path: str, **kwargs):
    # [FUNCTION DOC] Loads a multimodal processor. Inkling checkpoints (detected lazily) use a
    # dedicated training processor from the plugin package. Otherwise uses the auto processor; load
    # errors (operating system or value errors) are logged and give none. A result that is a bare
    # tokenizer or not a processor is discarded (none). Returns the processor or none.
    from miles.utils.chat_template_utils.inkling import is_inkling_checkpoint

    if is_inkling_checkpoint(name_or_path):
        from miles_plugins.models.inkling.mm_processor import InklingTrainProcessor

        return InklingTrainProcessor(name_or_path)

    try:
        proc = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except (OSError, ValueError) as e:
        logger.warning(f"Failed to load processor from {name_or_path}: {e}")
        proc = None

    # If HF returned a tokenizer, discard it.
    if isinstance(proc, PreTrainedTokenizerBase) or not isinstance(proc, ProcessorMixin):
        proc = None

    return proc


def process_vision_info(prompt, processor):
    # [FUNCTION DOC] Extracts media from a chat prompt. If the processor can extract media itself,
    # delegates to it. Otherwise uses the qwen vision utility with the image processor's patch size
    # (default 14, logged when used; the qwen3 models use 16) and returns a mapping with images and
    # videos entries. The default is a temporary choice (see the to-do).
    if hasattr(processor, "extract_media"):
        return processor.extract_media(prompt)

    # TODO: temporary solution, will write image utils for miles later
    from qwen_vl_utils import process_vision_info as qwen_process_vision_info

    if hasattr(processor.image_processor, "patch_size"):
        image_patch_size = processor.image_processor.patch_size
    else:
        logger.info(f"Using default patch size: {DEFAULT_PATCH_SIZE}")
        image_patch_size = DEFAULT_PATCH_SIZE
    images, videos = qwen_process_vision_info(prompt, image_patch_size=image_patch_size)
    multimodal_inputs = {"images": images, "videos": videos}
    return multimodal_inputs


def encode_image_for_rollout_engine(image) -> str:
    # [FUNCTION DOC] Converts a PIL-style image to RGB if needed, saves it as png into memory and
    # returns a data url with base64 text.
    # KNOWN ISSUE: the docstring says the image is loaded from a path, but the function takes an
    # already loaded image object.
    """Load an image from path, ensure RGB, encode as PNG base64 string."""
    buffer = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"
