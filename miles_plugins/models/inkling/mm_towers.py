# [MODULE DOC] Multimodal tower modules for Inkling and the logic that wires them into the language
# model's forward. Towers: an audio encoder (embedding table over mel-bin and bin-value pairs, summed
# over mel bins, optional RMS norm) and a vision encoder wrapper around the hierarchical-MLP patch
# encoder. Helpers load tower weights from Hugging Face safetensors shards and construct the towers
# (frozen unless training), and a wiring function wraps the model's forward so that, on the first
# pipeline stage, tower outputs replace the embeddings at given token positions before the decoder runs.
from __future__ import annotations

import json

import torch
import torch.nn as nn

from miles_plugins.models.inkling.vision_encoder import HMLPPatchEncoder, InklingVisionConfig, RMSNorm


class InklingAudioEncoder(nn.Module):
    # [CLASS DOC] Discrete-mel audio encoder: embeds each (mel bin index, quantized value) pair from a
    # single table of mel bins times vocabulary rows, sums the embeddings over mel bins per frame, and
    # optionally RMS-normalizes. Output width is the decoder model dimension.
    def __init__(self, audio_cfg: dict):
        # [FUNCTION DOC] Asserts the audio mode is dmel (default). Reads mel bin count, vocabulary
        # size and decoder width from the mapping (missing keys raise key errors); the norm is built
        # unless use_audio_norm is false (epsilon 1e-6).
        super().__init__()
        assert audio_cfg.get("audio_mode", "dmel") == "dmel"
        self.n_mel_bins = audio_cfg["n_mel_bins"]
        self.mel_vocab_size = audio_cfg["mel_vocab_size"]
        self.encoder = nn.Embedding(self.n_mel_bins * self.mel_vocab_size, audio_cfg["decoder_dmodel"])
        self.final_norm = (
            RMSNorm(audio_cfg["decoder_dmodel"], eps=1e-6) if audio_cfg.get("use_audio_norm", True) else None
        )

    def forward(self, dmel: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Input (frames, mel bins) integer bin values; asserts the mel dimension. Row
        # index per element is mel position times vocabulary size plus the value; looks up embeddings,
        # sums over the mel axis, and applies the norm if present. Returns (frames, decoder width).
        assert dmel.shape[1] == self.n_mel_bins
        idx = (torch.arange(self.n_mel_bins, device=dmel.device) * self.mel_vocab_size).unsqueeze(0) + dmel.to(
            torch.int32
        )
        h = self.encoder(idx.reshape(-1)).reshape(dmel.shape[0], self.n_mel_bins, -1).sum(axis=1)
        return self.final_norm(h) if self.final_norm is not None else h


class InklingVisionEncoder(nn.Module):
    # [CLASS DOC] Thin wrapper holding the hierarchical-MLP patch encoder as submodule vision_encoder
    # (so checkpoint keys match).
    def __init__(self, vision_cfg: dict):
        # [FUNCTION DOC] Asserts the encoder type is hmlp (default) and builds the encoder from the
        # config mapping.
        super().__init__()
        assert vision_cfg.get("vision_encoder_type", "hmlp") == "hmlp"
        self.vision_encoder = HMLPPatchEncoder(InklingVisionConfig(vision_cfg))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Returns the patch encoder output, one embedding row per input patch.
        return self.vision_encoder(patches)


def _load_tower_tensors(hf_checkpoint: str) -> dict[str, torch.Tensor]:
    # [FUNCTION DOC] HELPER: Reads the checkpoint's weight index, selects names starting with visual. or
    # audio. (optionally prefixed with model.), groups them by shard file, opens each shard once and
    # loads those tensors to host. Returns a mapping from the normalized name (model. prefix removed)
    # to tensor. File I/O; the index file handle is left to garbage collection.
    """Read tower tensors from HF safetensors, normalizing keys to visual.* / audio.*."""
    from safetensors import safe_open

    index = json.load(open(f"{hf_checkpoint}/model.safetensors.index.json"))
    by_file: dict[str, list[str]] = {}
    for name, fname in index["weight_map"].items():
        bare = name[len("model.") :] if name.startswith(("model.visual.", "model.audio.")) else name
        if bare.startswith(("visual.", "audio.")):
            by_file.setdefault(fname, []).append(name)
    out: dict[str, torch.Tensor] = {}
    for fname, names in by_file.items():
        with safe_open(f"{hf_checkpoint}/{fname}", framework="pt") as f:
            for name in names:
                bare = name[len("model.") :] if name.startswith(("model.visual.", "model.audio.")) else name
                out[bare] = f.get_tensor(name)
    return out


def build_mm_towers(
    hf_checkpoint: str, device, dtype, train: bool = False
) -> tuple[nn.Module | None, nn.Module | None]:
    # [FUNCTION DOC] Reads config.json and tower tensors. Builds the visual tower if the config has a
    # vision section and visual weights exist, loading them strictly (prefix stripped); same for audio.
    # Each built tower is moved to the device and dtype, gradients enabled only if training, and put in
    # eval mode when not training. Returns (visual, audio) with none for absent modalities.
    """Construct + HF-load both towers; None when the checkpoint has no config for a modality."""
    cfg = json.load(open(f"{hf_checkpoint}/config.json"))
    weights = _load_tower_tensors(hf_checkpoint)

    visual = audio = None
    if cfg.get("vision_config") and any(k.startswith("visual.") for k in weights):
        visual = InklingVisionEncoder(cfg["vision_config"])
        sd = {k[len("visual.") :]: v for k, v in weights.items() if k.startswith("visual.")}
        visual.vision_encoder.load_state_dict(sd, strict=True)
    if cfg.get("audio_config") and any(k.startswith("audio.") for k in weights):
        audio = InklingAudioEncoder(cfg["audio_config"])
        sd = {k[len("audio.") :]: v for k, v in weights.items() if k.startswith("audio.")}
        audio.load_state_dict(sd, strict=True)

    for tower in (visual, audio):
        if tower is not None:
            tower.to(device=device, dtype=dtype)
            tower.requires_grad_(train)
            if not train:
                tower.eval()
    return visual, audio


def wire_mm_towers(model, hf_checkpoint: str, train: bool = False) -> None:
    # [FUNCTION DOC] Replaces the model's forward (mutates the model). Stages that are not first in the
    # pipeline get a pass-through that swallows the six multimodal keyword arguments and calls the
    # original. The first stage builds the towers on the current GPU in the model's parameter dtype,
    # stores them in the model's raw attribute dictionary (so they are not registered submodules and
    # are not in its state dict or parameters), and installs a forward that: validates that positions
    # and features have equal counts; computes text embeddings; runs the vision and/or audio tower on
    # the features (with gradients only when training) and scatters results into the embeddings at the
    # given positions; passes the result as the decoder input to the original forward.
    """Wrap an InklingGPTModel's forward for multimodal training (pre_process builds+merges;
    later stages pass through). train=True trains the towers instead of freezing them
    (weight-sync/ckpt for tower params is not wired yet)."""
    import megatron.core.parallel_state as ps

    _orig_forward = model.forward

    if not getattr(model, "pre_process", False):

        def _mm_passthrough(
            *a,
            mm_vision_patches=None,
            mm_vision_positions=None,
            mm_vision_num_patches=None,
            mm_audio_dmel=None,
            mm_audio_positions=None,
            mm_audio_num_tokens=None,
            **kw,
        ):
            # [FUNCTION DOC] HELPER: Ignores the six multimodal keyword arguments and calls the
            # original forward with the remaining arguments.
            return _orig_forward(*a, **kw)

        model.forward = _mm_passthrough
        return

    device = torch.cuda.current_device()
    visual, audio = build_mm_towers(hf_checkpoint, device=device, dtype=model.config.params_dtype, train=train)
    model.__dict__["_mm_towers"] = (visual, audio)

    def _scatter(decoder_input, embeds, positions):
        # [FUNCTION DOC] HELPER: Returns the decoder input unchanged if there are no embeddings or
        # positions. Under sequence parallel with tensor parallel size above one, positions are global
        # and decoder_input holds this rank's sequence shard, so positions are shifted by rank times
        # shard length and only those inside the shard (and their embeddings) are kept. Clones the
        # decoder input (sequence, batch, hidden) and writes the embeddings (cast to its dtype) at the
        # selected sequence positions in batch index 0. Returns the new tensor.
        if embeds is None or positions is None or positions.numel() == 0:
            return decoder_input
        s_local = decoder_input.shape[0]
        if model.config.sequence_parallel and ps.get_tensor_model_parallel_world_size() > 1:
            rank = ps.get_tensor_model_parallel_rank()
            local = positions - rank * s_local
            sel = (local >= 0) & (local < s_local)
            local, emb = local[sel], embeds[sel]
        else:
            local, emb = positions, embeds
        decoder_input = decoder_input.clone()
        if local.numel():
            decoder_input[local, 0] = emb.to(decoder_input.dtype)
        return decoder_input

    def _mm_forward(
        *a,
        mm_vision_patches=None,
        mm_vision_positions=None,
        mm_vision_num_patches=None,
        mm_audio_dmel=None,
        mm_audio_positions=None,
        mm_audio_num_tokens=None,
        **kw,
    ):
        # [FUNCTION DOC] HELPER: Replacement forward for the first pipeline stage. Takes input ids from
        # the keyword or first positional argument and position ids from the keyword. Asserts that each
        # provided feature tensor has a matching positions tensor with the same row count. Computes
        # embeddings; vision features (moved to the embeddings' device) go through the visual tower and
        # audio frames through the audio tower, under gradient mode only when training; outputs are
        # scattered into the embeddings at their positions. Sets the decoder input keyword and calls the
        # original forward. The per-image patch counts and per-clip token counts are unused here.
        # KNOWN ISSUE: if a feature is provided but the corresponding tower is absent (none), calling
        # it fails with a type error.
        input_ids = kw.get("input_ids", a[0] if a else None)
        position_ids = kw.get("position_ids")
        if mm_vision_patches is not None:
            assert mm_vision_positions is not None, "mm_vision_patches given without mm_vision_positions"
            assert mm_vision_positions.numel() == mm_vision_patches.shape[0], (
                f"{mm_vision_positions.numel()} vision position(s) vs " f"{mm_vision_patches.shape[0]} patch row(s)"
            )
        if mm_audio_dmel is not None:
            assert mm_audio_positions is not None, "mm_audio_dmel given without mm_audio_positions"
            assert mm_audio_positions.numel() == mm_audio_dmel.shape[0], (
                f"{mm_audio_positions.numel()} audio position(s) vs " f"{mm_audio_dmel.shape[0]} dmel frame(s)"
            )
        decoder_input = model.embedding(input_ids=input_ids, position_ids=position_ids)
        # the grad context must cover ONLY the tower forwards: _scatter clones
        # decoder_input, and a clone under no_grad detaches the embedding graph
        ctx = torch.enable_grad() if train else torch.no_grad()
        if mm_vision_patches is not None:
            with ctx:
                v_emb = model.__dict__["_mm_towers"][0](mm_vision_patches.to(device=decoder_input.device))
            decoder_input = _scatter(decoder_input, v_emb, mm_vision_positions.to(decoder_input.device))
        if mm_audio_dmel is not None:
            with ctx:
                a_emb = model.__dict__["_mm_towers"][1](mm_audio_dmel.to(device=decoder_input.device))
            decoder_input = _scatter(decoder_input, a_emb, mm_audio_positions.to(decoder_input.device))
        kw["decoder_input"] = decoder_input
        return _orig_forward(*a, **kw)

    model.forward = _mm_forward
