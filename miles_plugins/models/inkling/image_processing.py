# [MODULE DOC] Image preprocessing for the Inkling vision tower without Hugging Face dependencies.
# Images are optionally upscaled (long edge doubled, capped at 2048 unless already larger), cut into
# square patches (rows rounded up, columns floor(width/patch)+1, so there is always one padded
# column), normalized per channel with CLIP mean/std, and out-of-image cells are filled with a
# normalized pad value (-1/255 raw). Output patches are bfloat16 with a size-2 axis (the same patch
# repeated, as an expanded view) in the layout (patches, 2, height, width, channels). An optional
# numba kernel computes the patches; otherwise a numpy version with identical math is used.
from __future__ import annotations

import io
import math

import numpy as np
import torch

IMAGE_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
IMAGE_STD = np.array([0.26862954, 0.2613026, 0.2757771], dtype=np.float32)
PAD_RAW_VALUE = np.float32(-1.0 / 255.0)
PAD_NORM = (np.full((3,), PAD_RAW_VALUE, dtype=np.float32) - IMAGE_MEAN) / IMAGE_STD

RESCALE_IMAGE_FRAC = 2.0
RESCALE_IMAGE_MAX_UPSCALED_LONG_EDGE = 2048


def _scaled_image_dimensions(
    width: int,
    height: int,
    rescale_image_frac: float | None,
    rescale_image_max_upscaled_long_edge: int | None,
) -> tuple[int, int]:
    # [FUNCTION DOC] HELPER: Returns (width, height) after rescaling. No scaling when the fraction is
    # none, the long edge is zero, or the resulting ratio is exactly one. Target long edge is long edge
    # times fraction, capped at max(cap, long edge) when a cap is given (so images already larger than
    # the cap are never shrunk). Each side is scaled by the ratio, rounded half up, minimum 1.
    if rescale_image_frac is None:
        return width, height
    long_edge = max(width, height)
    if long_edge == 0:
        return width, height
    target_long_edge = float(long_edge) * rescale_image_frac
    if rescale_image_max_upscaled_long_edge is not None:
        effective_cap = max(rescale_image_max_upscaled_long_edge, long_edge)
        target_long_edge = min(target_long_edge, float(effective_cap))
    ratio = target_long_edge / float(long_edge)
    if ratio == 1.0:
        return width, height

    def scale(value: int) -> int:
        # [FUNCTION DOC] HELPER: Scales one side by the ratio, rounding half up, at least 1.
        return max(1, math.floor(float(value) * ratio + 0.5))

    return scale(width), scale(height)


def _load_image_bytes(image) -> bytes:
    # [FUNCTION DOC] HELPER: Bytes-like input is returned as bytes. Strings: URLs and data URIs raise
    # a value error (must be resolved upstream); otherwise a path (file:// prefix stripped) read from
    # disk. Other inputs (imaging-library image, tensor, or array) are converted: tensors move to host
    # and are cast to unsigned bytes, arrays become images, non-RGB images are converted to RGB, then
    # encoded as PNG in memory. Returns the bytes.
    """Coerce a single image input into raw PNG/JPEG bytes for preprocessing."""
    if isinstance(image, (bytes, bytearray, memoryview)):
        return bytes(image)
    if isinstance(image, str):
        if image.startswith(("http://", "https://", "data:")):
            raise ValueError(
                "Inkling image patchifier received a URL/data: image; resolve it to "
                "bytes upstream before preprocessing."
            )
        path = image[len("file://") :] if image.startswith("file://") else image
        with open(path, "rb") as f:
            return f.read()

    from PIL import Image

    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().numpy()
        image = Image.fromarray(arr.astype("uint8") if arr.dtype != np.uint8 else arr)
    elif not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _fill_patches_numpy(
    arr: np.ndarray,
    patch_size: int,
    patches: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    pad_norm: np.ndarray,
) -> None:
    # [FUNCTION DOC] HELPER: Fills the preallocated output (patch count, size, size, 3) in place.
    # Builds a float32 canvas of whole patches pre-filled with the pad value, normalizes the image as
    # raw times 1/255, minus mean, over std, copies it into the top-left (truncating columns beyond the
    # canvas), then reshapes into row-major patches. Patch rows = ceil(height / size), patch columns =
    # floor(width / size) + 1.
    """Same math as the serving-side numba kernel: normalize in-image pixels,
    pad-fill out-of-image cells; float32 throughout (bit-equal op order:
    raw*inv255 -> -mean -> /std per channel)."""
    h, w = arr.shape[0], arr.shape[1]
    nph = (h + patch_size - 1) // patch_size
    npw = w // patch_size + 1
    inv255 = np.float32(1.0 / 255.0)

    padded_h, padded_w = nph * patch_size, npw * patch_size
    canvas = np.empty((padded_h, padded_w, 3), dtype=np.float32)
    canvas[:] = pad_norm
    norm = (arr.astype(np.float32) * inv255 - mean) / std
    canvas[:h, : min(w, padded_w)] = norm[:, : min(w, padded_w)]

    view = canvas.reshape(nph, patch_size, npw, patch_size, 3).transpose(0, 2, 1, 3, 4)
    patches[:] = view.reshape(nph * npw, patch_size, patch_size, 3)


try:
    from numba import njit

    @njit(cache=True)
    def _fill_patches_numba(arr, patch_size, patches, mean, std, pad_norm):  # pragma: no cover
        # [FUNCTION DOC] HELPER: Compiled (cached) loop version of the same fill: for each patch index,
        # for each pixel, writes the normalized value if the pixel lies inside the image, else the pad
        # value, per channel. Writes into the output in place.
        h = arr.shape[0]
        w = arr.shape[1]
        nph = (h + patch_size - 1) // patch_size
        npw = w // patch_size + 1
        inv255 = np.float32(1.0 / 255.0)

        for k in range(nph * npw):
            i = k // npw
            j = k - i * npw
            y_base = i * patch_size
            x_base = j * patch_size

            for y in range(patch_size):
                iy = y_base + y
                for x in range(patch_size):
                    ix = x_base + x
                    if iy < h and ix < w:
                        for c in range(3):
                            raw = np.float32(arr[iy, ix, c]) * inv255
                            patches[k, y, x, c] = (raw - mean[c]) / std[c]
                    else:
                        for c in range(3):
                            patches[k, y, x, c] = pad_norm[c]

    _fill_patches = _fill_patches_numba
except ImportError:  # pragma: no cover
    _fill_patches = _fill_patches_numpy


def encode_image_bytes(image_bytes: bytes, patch_size: int) -> torch.Tensor:
    # [FUNCTION DOC] Raises a value error for non-positive patch size. Decodes the image to RGB,
    # rescales (Lanczos resampling) per the module constants when the size changes, converts to an
    # unsigned byte array, fills a float32 patch array, and returns a bfloat16 tensor of shape
    # (patches, 2, size, size, 3) where the size-2 axis is an expanded (non-copied) repeat.
    if patch_size <= 0:
        raise ValueError("patch_size must be greater than zero")

    from PIL import Image

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    scaled_size = _scaled_image_dimensions(
        *image.size,
        rescale_image_frac=RESCALE_IMAGE_FRAC,
        rescale_image_max_upscaled_long_edge=RESCALE_IMAGE_MAX_UPSCALED_LONG_EDGE,
    )
    if scaled_size != image.size:
        image = image.resize(scaled_size, resample=Image.Resampling.LANCZOS)
    arr = np.array(image, dtype=np.uint8, copy=True)
    height, width, _ = arr.shape

    nph = (height + patch_size - 1) // patch_size
    npw = width // patch_size + 1
    num_patches = nph * npw

    patches = np.empty((num_patches, patch_size, patch_size, 3), dtype=np.float32)
    _fill_patches(arr, patch_size, patches, IMAGE_MEAN, IMAGE_STD, PAD_NORM)

    return (
        torch.from_numpy(patches)
        .to(torch.bfloat16)
        .view(num_patches, 1, patch_size, patch_size, 3)
        .expand(num_patches, 2, patch_size, patch_size, 3)
    )


class InklingImagePatchifier:
    # [CLASS DOC] Converts images to patch tensors with a fixed patch size (default 40).
    """images -> {vision_patches_bthwc, num_patches, num_tokens} (dict, no HF deps)."""

    def __init__(self, patch_size: int = 40):
        # [FUNCTION DOC] Stores the patch size.
        self.patch_size = patch_size

    def preprocess(self, images: list | object) -> dict:
        # [FUNCTION DOC] Accepts one image or a list/tuple; each is loaded to bytes and encoded.
        # Returns a mapping: "vision_patches_bthwc" (all images' patches concatenated along dim 0; an
        # empty 1-D tensor if no images), "num_patches" (per-image patch counts) and "num_tokens" (a
        # copy of the counts; one token per patch).
        if not isinstance(images, (list, tuple)):
            images = [images]

        per_image_patches: list[torch.Tensor] = []
        num_patches: list[int] = []
        for img in images:
            vp = encode_image_bytes(_load_image_bytes(img), patch_size=self.patch_size)
            per_image_patches.append(vp)
            num_patches.append(int(vp.shape[0]))

        if len(per_image_patches) == 1:
            vision_patches_bthwc = per_image_patches[0]
        elif per_image_patches:
            vision_patches_bthwc = torch.cat(per_image_patches, dim=0)
        else:
            vision_patches_bthwc = torch.empty(0)

        return {
            "vision_patches_bthwc": vision_patches_bthwc,
            "num_patches": num_patches,
            "num_tokens": list(num_patches),
        }
