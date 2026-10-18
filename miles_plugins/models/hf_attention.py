# [MODULE DOC] Base class for attention layers that delegate computation to a Hugging Face style
# implementation (for example linear attention) inside Megatron. Handles sequence-parallel gathering,
# context-parallel relayout (either native "hybrid" CP where the inner module does its own state
# passing, or a gather-everything fallback that runs the inner module on the full sequence and
# re-slices into the zigzag layout), and the batch-major permutation the inner module expects. Also
# provides a custom all-gather for duplicated computation and a helper that configures hybrid CP.
import logging
from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
import torch.nn as nn
from megatron.core import mpu, tensor_parallel
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import MegatronModule

from miles.utils.hf_utils.config import load_hf_config
from miles_plugins.models.cp_utils import packed_shard_to_zigzag, zigzag_to_packed_shard

logger = logging.getLogger(__name__)


class _AllGatherForDuplicatedComputation(torch.autograd.Function):
    # [CLASS DOC] HELPER: Autograd function: forward all-gathers a tensor over a group and returns a
    # tuple with one tensor per rank; backward returns only the gradient of this rank's own entry
    # (no cross-rank reduction), correct when downstream computation is duplicated on every rank.
    """All-gather whose backward just returns the local gradient slice (no reduce).

    Use this instead of ``dist.nn.all_gather`` when the computation after the
    gather is *duplicated* across ranks (same weights, same full input ->
    identical gradients). The default ``all_gather`` backward performs a
    reduce-scatter, which incorrectly sums ``world_size`` identical copies of
    the gradient.
    """

    @staticmethod
    def forward(ctx, x, group):
        # [FUNCTION DOC] HELPER: Saves group and rank on the context, allocates one empty tensor per
        # rank, runs a blocking all-gather of the contiguous input, returns the tuple of gathered tensors.
        ctx.group = group
        ctx.rank = dist.get_rank(group=group)
        out = [torch.empty_like(x) for _ in range(dist.get_world_size(group=group))]
        dist.all_gather(out, x.contiguous(), group=group)
        return tuple(out)

    @staticmethod
    def backward(ctx, *grads):
        # [FUNCTION DOC] HELPER: Returns this rank's gradient entry for the input and none for group.
        return grads[ctx.rank], None


class HuggingfaceAttention(MegatronModule, ABC):
    # [CLASS DOC] Abstract Megatron attention module. Subclasses implement the hook that maps
    # batch-major hidden states (batch, sequence, hidden) plus packed sequence parameters to an
    # output of the same layout. The class flag hybrid_cp (default false) says the inner module
    # handles context parallelism natively.
    """Attention layer abstract class.

    This layer only contains common modules required for the "self attn" and
    "cross attn" specializations.
    """

    # Subclasses set this to True when the underlying module handles CP natively
    # (e.g. via fla's state-passing CP for DeltaNet), bypassing the all-gather.
    hybrid_cp: bool = False

    def __init__(
        self,
        args,
        config,
        layer_number: int,
        cp_comm_type: str = "p2p",
        pg_collection=None,
        name: str | None = None,
    ):
        # [FUNCTION DOC] Stores args, config, the one-based Megatron layer number and the zero-based
        # Hugging Face layer index; loads the HF config from the checkpoint path in args (file I/O) and
        # sets its attention implementation to flash attention 2 (hard coded). The communication type,
        # process-group collection and name arguments are unused.
        super().__init__(config=config)
        self.args = args
        self.config = config
        # Note that megatron layer_number starts at 1
        self.layer_number = layer_number
        self.hf_layer_idx = layer_number - 1
        self.hf_config = load_hf_config(args.hf_checkpoint)
        # hardcode to fa2 at the moment.
        self.hf_config._attn_implementation = "flash_attention_2"

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        inference_context: BaseInferenceContext | None = None,
        rotary_pos_emb: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        rotary_pos_cos: torch.Tensor | None = None,
        rotary_pos_sin: torch.Tensor | None = None,
        rotary_pos_cos_sin: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: int | None = None,
        *,
        inference_params: BaseInferenceContext | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # [FUNCTION DOC] Input (sequence, batch, hidden) with required packed sequence parameters
        # (assertion); only the query cumulative lengths are used, all other arguments ignored.
        # (1) With sequence parallel: gather the sequence over the tensor-parallel group, with the
        # backward splitting rather than reduce-scattering (inner module is duplicated). (2) With
        # context parallel size above one and hybrid CP: unless the all-gather-CP data layout is in
        # use (already contiguous shards), convert from zigzag to packed contiguous shards. (3) With CP
        # and not hybrid: all-gather every rank's hidden states (duplicated-computation gather), then
        # per sequence reassemble the global order: for each sequence take, in rank order, each rank's
        # first half-chunk, followed by the second half-chunks in reverse rank order, and concatenate.
        # (4) Permute to batch-major, call the hook, permute back. (5) Reverse the CP relayout:
        # hybrid converts packed shards back to zigzag (unless all-gather CP); non-hybrid slices each
        # sequence into twice cp-size chunks and keeps chunk rank and chunk (2*cp-1-rank). (6) With
        # sequence parallel, scatter back along the sequence. Returns (output, none).
        # KNOWN ISSUE: the non-hybrid path assumes every sequence length is divisible by twice the cp
        # size and takes lengths from tensor elements (host synchronization per sequence); a todo in
        # the code notes this. An unused local (chunk size) is computed in the output re-slicing loop.
        assert packed_seq_params is not None
        cu_seqlens = packed_seq_params.cu_seqlens_q

        if self.args.sequence_parallel:
            # tensor_parallel_output_grad=False: the linear attention after this
            # gather is NOT TP-sharded (duplicated on all ranks), so the backward
            # should split (not reduce-scatter) to avoid inflating gradients by TP.
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states,
                tensor_parallel_output_grad=False,
                group=mpu.get_tensor_model_parallel_group(),
            )

        if mpu.get_context_parallel_world_size() > 1 and self.hybrid_cp:
            cp_size = mpu.get_context_parallel_world_size()
            # Native fla CP expects each rank to own a contiguous shard of the
            # packed global token stream. In allgather-CP mode the data pipeline
            # already provides that layout, so no extra relayout is
            # needed here.
            if not self.args.allgather_cp:
                hidden_states = zigzag_to_packed_shard(
                    hidden_states,
                    cu_seqlens,
                    mpu.get_context_parallel_group(),
                    mpu.get_context_parallel_rank(),
                    cp_size,
                )

        elif mpu.get_context_parallel_world_size() > 1:
            cp_size = mpu.get_context_parallel_world_size()
            # Use custom all-gather whose backward returns local gradient
            # instead of reduce-scatter, since the computation is duplicated.
            hidden_states_list = _AllGatherForDuplicatedComputation.apply(
                hidden_states,
                mpu.get_context_parallel_group(),
            )

            # TODO: preprocess this for each batch to prevent tolist in the training step
            whole_hidden_states_list = []

            local_cu_seqlens = cu_seqlens // cp_size
            for i in range(len(cu_seqlens) - 1):
                seqlen = cu_seqlens[i + 1] - cu_seqlens[i]
                chunk_size = seqlen // 2 // cp_size
                whole_hidden_states_list.extend(
                    [
                        hidden_states_list[cp_rank][local_cu_seqlens[i] : local_cu_seqlens[i] + chunk_size]
                        for cp_rank in range(cp_size)
                    ]
                    + [
                        hidden_states_list[cp_rank][local_cu_seqlens[i] + chunk_size : local_cu_seqlens[i + 1]]
                        for cp_rank in range(cp_size)
                    ][::-1],
                )
            hidden_states = torch.cat(whole_hidden_states_list, dim=0)

        hidden_states = hidden_states.permute(1, 0, 2)  # [bsz, seq_len, hidden_dim]

        output = self.hf_forward(hidden_states, packed_seq_params)
        bias = None

        output = output.permute(1, 0, 2)  # [seq_len, bsz, hidden_dim]

        if mpu.get_context_parallel_world_size() > 1 and self.hybrid_cp:
            if not self.args.allgather_cp:
                output = packed_shard_to_zigzag(
                    output,
                    cu_seqlens,
                    mpu.get_context_parallel_group(),
                    mpu.get_context_parallel_rank(),
                    cp_size,
                )

        elif mpu.get_context_parallel_world_size() > 1:
            cp_rank = mpu.get_context_parallel_rank()
            output_list = []
            for i in range(len(cu_seqlens) - 1):
                seqlen = cu_seqlens[i + 1] - cu_seqlens[i]
                chunk_size = seqlen // 2 // cp_size
                seq = output[cu_seqlens[i] : cu_seqlens[i + 1]]
                chunks = torch.chunk(seq, 2 * cp_size, dim=0)
                output_list.append(chunks[cp_rank])
                output_list.append(chunks[2 * cp_size - 1 - cp_rank])
            output = torch.cat(output_list, dim=0)

        if self.args.sequence_parallel:
            output = tensor_parallel.scatter_to_sequence_parallel_region(
                output, group=mpu.get_tensor_model_parallel_group()
            )

        return output, bias

    @abstractmethod
    def hf_forward(self, hidden_states, packed_seq_params):
        # [FUNCTION DOC] Abstract hook: takes batch-major hidden states and packed sequence parameters,
        # returns the attention output in the same layout.
        """Huggingface forward function"""


def detect_and_setup_hybrid_cp(model: nn.Module, cp_group: dist.ProcessGroup, cp_rank: int, cp_world_size: int) -> int:
    # [FUNCTION DOC] Walks all submodules; for each HF attention wrapper that has a linear-attention
    # submodule attribute, stores the cp group, rank and world size on that submodule and sets the
    # wrapper's hybrid flag true (instance attribute). Logs the count at info level when positive and
    # returns the count. Mutates modules.
    """Scan for GatedDeltaNet modules and configure them for native fla CP."""
    count = 0
    for module in model.modules():
        if isinstance(module, HuggingfaceAttention):
            linear_attn = getattr(module, "linear_attn", None)
            if linear_attn is not None:
                linear_attn.cp_group = cp_group
                linear_attn.cp_rank = cp_rank
                linear_attn.cp_world_size = cp_world_size
                module.hybrid_cp = True
                count += 1

    if count > 0:
        logger.info(f"Configured hybrid CP on {count} GDN modules (fla native state passing)")
    return count
