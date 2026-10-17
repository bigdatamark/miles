# [MODULE DOC]
# Docker build-time script (AMD/ROCm images) that compiles, ahead of time and without a GPU, the
# aiter JIT kernels the serving engine would otherwise compile at first use: the batched-prefill
# multi-head-attention module (two variants, with and without the causal mask for single-token
# extend) and the ragged paged-attention decode kernels over a grid of grouped-query ratios (1 to
# 16), head sizes (64 and 128) and partition loop counts (1 to 4). Environment: GPU_ARCHS must be
# exactly one target architecture name starting with gfx and without a semicolon (otherwise the
# process exits with a message); MAX_JOBS gives the thread count for parallel kernel builds. At
# import time it locates the installed aiter package directory, prepends it and its inner
# directory to the module search path, and imports aiter's JIT core by path (importing the aiter
# package itself would probe a GPU). Outputs: shared libraries in aiter's user JIT directory and
# build directory; intermediate build trees are deleted to keep the image small; every produced
# library is asserted to carry device code for exactly the target architecture. Prints size
# summaries. Entry point is main.
"""Prebuild the aiter JIT modules that sglang's aiter attention backend compiles on first use.

The base image builds aiter with PREBUILD_KERNELS=1, which leaves out the mha_batch_prefill
(extend) modules, and the pa_ragged (decode) kernels are only ever compiled at runtime. So every
fresh container spends minutes in hipcc before its first rollout. This covers bf16 models with a
bf16 KV cache and no ALiBi, logits soft-cap, sliding window or sinks; anything else still JITs.

The build host has no GPU, so GPU_ARCHS must name the target. `import aiter` probes the GPU, so
this loads aiter's JIT core by path, the way aiter's own setup.py does.
"""

import ast
import concurrent.futures
import importlib.util
import os
import shutil
import sys
import types

import torch

AITER_DIR = os.path.dirname(importlib.util.find_spec("aiter").submodule_search_locations[0])
sys.path[:0] = [os.path.join(AITER_DIR, "aiter"), AITER_DIR]

from csrc.cpp_itfs import utils as itfs_utils  # noqa: E402
from csrc.cpp_itfs.pa import pa_ragged  # noqa: E402
from jit import core  # noqa: E402

# Decode shapes: GQA ratio x head size x ceil(context / (256 * 64)). npar_loops <= 4 covers
# contexts up to 64K, which every model the nightly runs fits in.
GQA_RATIOS = range(1, 17)
HEAD_SIZES = (64, 128)
NPAR_LOOPS = range(1, 5)


def _check_archs(so_path, gfx):
    # [FUNCTION DOC] HELPER: Asserts that the shared library at the path embeds device code for
    # exactly the single architecture gfx (using aiter's offload-architecture reader); raises an
    # assertion error listing the found architectures otherwise. aiter would silently rebuild a
    # module whose device code lacks the running GPU's architecture.
    # aiter rebuilds a module whose device code does not include the running GPU's arch.
    archs = core._so_offload_archs(so_path)
    assert archs == {gfx}, f"{so_path} targets {sorted(archs)}, not {gfx}"


def _mha_batch_prefill_gen():
    # [FUNCTION DOC] HELPER: Obtains aiter's own command generator function for the batched
    # prefill attention module without importing the aiter package. Parses aiter's mha operations
    # source file into a syntax tree, extracts the single function definition with that name
    # (assertion on unpacking if not exactly one), clears its decorators, compiles it alone and
    # executes it in a minimal namespace providing the torch module, tensor and generator types, a
    # dtype namespace (bf16 and fp8 e4m3 fnuz) and aiter's CK directory. Returns the function.
    # Side effect: reads a file and executes code derived from it.
    """aiter's own cmdGenFunc_mha_batch_prefill, lifted out of aiter/ops/mha.py, which imports aiter."""
    path = os.path.join(AITER_DIR, "aiter", "ops", "mha.py")
    with open(path) as f:
        tree = ast.parse(f.read())
    (fn,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "cmdGenFunc_mha_batch_prefill"]
    fn.decorator_list = []
    scope = {
        "torch": torch,
        "Tensor": torch.Tensor,
        "Generator": torch.Generator,
        "dtypes": types.SimpleNamespace(bf16=torch.bfloat16, fp8=torch.float8_e4m3fnuz),
        "CK_DIR": core.CK_DIR,
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), path, "exec"), scope)
    return scope[fn.name]


def build_mha_batch_prefill(gfx):
    # [FUNCTION DOC] Builds the batched prefill attention JIT module for the target architecture.
    # Runs twice, for maximum query length 2 (causal mask) and 1 (a one-token extend, which drops
    # the causal mask), producing two module variants. Each time: starts from aiter's default build
    # arguments for the module, overlays the generator's output computed with empty bf16 tensors
    # and sglang's call parameters (key length 4096, no dropout, scale 1.0, no soft cap, causal,
    # no sliding window, no sink, no softmax output), builds the module with those arguments,
    # deletes the module's build tree (keeping only the shared library), asserts the library
    # targets exactly the architecture, and prints its size in mebibytes. Long compile; writes into
    # aiter's JIT directories.
    gen = _mha_batch_prefill_gen()
    q = torch.empty(0, dtype=torch.bfloat16)
    # sglang's extend call; a one-token extend (max_seqlen_q=1) drops the causal mask.
    for max_seqlen_q in (2, 1):
        args = core.get_args_of_build("module_mha_batch_prefill")
        args.update(
            gen(
                q, q, q, None, None, None,
                max_seqlen_q=max_seqlen_q, max_seqlen_k=4096, dropout_p=0.0, softmax_scale=1.0,
                logits_soft_cap=0.0, zero_tensors=False, is_causal=True, window_size_left=-1,
                window_size_right=-1, sink_size=0, return_softmax_lse=False, return_dropout_randval=False,
            )
        )  # fmt: skip
        md_name = args["md_name"]
        core.build_module(
            md_name,
            args["srcs"],
            args["flags_extra_cc"],
            args["flags_extra_hip"],
            args["blob_gen_cmd"],
            args["extra_include"],
            args["extra_ldflags"],
            args["verbose"],
            args["is_python_module"],
            args["is_standalone"],
            args["torch_exclude"],
            args.get("third_party", []),
            args.get("hipify", False),
            flags_extra_hip_per_source=args.get("flags_extra_hip_per_source", {}),
        )
        # The build tree holds the generated blobs and objects; only the .so is loaded.
        shutil.rmtree(os.path.join(core.bd_dir, md_name))
        so_path = os.path.join(core.get_user_jit_dir(), f"{md_name}.so")
        _check_archs(so_path, gfx)
        print(f"[aiter] {md_name}.so: {os.path.getsize(so_path) / 2**20:.0f} MiB", flush=True)


def _mp_lock(lock_path, main_func, final_func=None, wait_func=None):
    # [FUNCTION DOC] HELPER: Replacement for aiter's multi-process build lock (whose stock version
    # imports the aiter package). Because each build here uses its own folder there is no
    # contention, so it simply calls the main function and then the final function if given.
    # KNOWN ISSUE: the lock path and wait function parameters are unused (kept for signature
    # compatibility).
    # The stock lock imports the aiter package. Every build here has its own folder, so none contend.
    main_func()
    if final_func is not None:
        final_func()


def _run_lib(func_name, folder=None):
    # [FUNCTION DOC] HELPER: Replacement for aiter's library loader (which would load the built
    # library and needs the GPU runtime). Returns the folder if given, otherwise the function name,
    # so the compile step reports the build folder without loading anything.
    # Stands in for the dlopen of the result, which needs the GPU runtime.
    return folder or func_name


def build_pa_ragged(gfx, jobs):
    # [FUNCTION DOC] Builds all ragged paged-attention decode kernels for the architecture using a
    # thread pool with the given number of workers. Side effect: replaces the lock and library
    # loader functions in aiter's cpp interface utilities module with the stand-ins above (global
    # monkey patch). One build per combination of grouped-query ratio, head size and partition loop
    # count (see module constants). After all builds, asserts each produced library targets exactly
    # the architecture, sums their sizes and prints the count, build directory and total size in
    # mebibytes. Any build error propagates from the pool.
    itfs_utils.mp_lock = _mp_lock
    itfs_utils.run_lib = _run_lib

    def build(shape):
        # [FUNCTION DOC] HELPER: Compiles one kernel for a (grouped-query ratio, head size, loop
        # count) triple using bf16 for query, key/value and output types, automatic KV cache dtype
        # and the argument values sglang's ragged paged-attention call uses (these key the build).
        # Then prunes the build folder, deleting every subdirectory (copied CK headers) and every
        # file except the shared library. Returns the folder name.
        gqa_ratio, head_size, npar_loops = shape
        # The argument values and types match sglang's paged_attention_ragged call, which keys the build.
        folder = pa_ragged.compile(
            gqa_ratio, head_size, npar_loops,
            "__hip_bfloat16", "__hip_bfloat16", "auto", "__hip_bfloat16",
            1, False, 256, 1, False,
        )  # fmt: skip
        build_dir = os.path.join(itfs_utils.BUILD_DIR, folder)
        # Each build copies the CK headers into its folder; only lib.so is loaded.
        for entry in os.listdir(build_dir):
            path = os.path.join(build_dir, entry)
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif entry != "lib.so":
                os.remove(path)
        return folder

    shapes = [(g, h, n) for g in GQA_RATIOS for h in HEAD_SIZES for n in NPAR_LOOPS]
    with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
        folders = list(pool.map(build, shapes))

    total = 0
    for folder in folders:
        lib = os.path.join(itfs_utils.BUILD_DIR, folder, "lib.so")
        _check_archs(lib, gfx)
        total += os.path.getsize(lib)
    print(f"[aiter] {len(folders)} pa_ragged kernels in {itfs_utils.BUILD_DIR}: {total / 2**20:.0f} MiB", flush=True)


def main():
    # [FUNCTION DOC] Entry point. Reads GPU_ARCHS and exits with a message unless it starts with
    # gfx and contains no semicolon (exactly one architecture). Builds the batched prefill module,
    # then lets aiter size the ninja worker count for the host (which sets MAX_JOBS), then builds
    # the paged-attention kernels with MAX_JOBS threads (a missing or non-integer MAX_JOBS raises
    # a key or value error).
    gfx = os.environ.get("GPU_ARCHS", "")
    if not gfx.startswith("gfx") or ";" in gfx:
        sys.exit(f"set GPU_ARCHS to the one target arch (got {gfx!r})")
    build_mha_batch_prefill(gfx)
    # Sizes MAX_JOBS to the host's cores and free memory.
    core.check_and_set_ninja_worker()
    build_pa_ragged(gfx, int(os.environ["MAX_JOBS"]))


if __name__ == "__main__":
    main()
