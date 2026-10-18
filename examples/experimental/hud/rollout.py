# [MODULE DOC]
# Custom rollout generate function and reward function for multi-turn, vision-language computer-use
# training on HUD environments hosted in cloud sandboxes. For each sample: mint the HUD task from
# the task row in the sample metadata, build the chat agent against the training run's inference
# router (using token-id-returning client), run the episode in a shared sandbox runtime under a
# process-wide concurrency cap and an episode timeout, then convert the graded trace into one
# training sample: concatenate the per-turn prompt and output token ids into one sequence with a
# loss mask (1 on model-emitted tokens, 0 elsewhere) and per-token rollout log probabilities,
# verifying that each later turn's prompt starts with the sequence so far (tolerating whitespace-only
# differences by re-synchronising), and rebuild the image tensors from the screenshots the policy
# saw, checked against the number of image placeholder tokens. Any failure (episode error, no
# usable turns, broken stitch, image count mismatch, no screenshot) yields a "written off" sample
# with status failed, removal flag set, zero response length and a one-image dummy so vision
# collectives stay aligned across ranks. Optional launch args read with defaults: Daytona key file,
# environment directory and snapshot name (required), sandbox cap (32), step budget (24),
# system prompt override, screenshot width (640), tokens per turn (512) and episode timeout seconds
# (1800). Module-level globals hold the shared runtime, its lock and the concurrency semaphore.
"""Multi-turn computer-use rollout on HUD v6 environments.

The division of labour, and why this file is small:

- **HUD** owns the episode: `task.run()` boots the environment (one Daytona
  sandbox per episode, deleted on exit), runs its agent loop against our
  policy, grades with the task template's second yield, and hands back a
  ``Trace`` whose ``AgentStep``s carry token-level ``Sample``s -- the exact
  prompt/output token ids and sampling logprobs of every turn, straight from
  the inference server (see sglang_compat.py for how stock sglang serves
  those).
- **This file** turns one graded HUD run into one Miles training sample:
  stitch the per-turn token ids into a single sequence with a loss mask, and
  rebuild the vision inputs (pixel_values) from the very screenshots the
  policy saw.

Stitching leans on a property verified per episode rather than assumed: the
server re-renders the chat template every turn, so turn k+1's prompt ids must
start with turn k's prompt+output ids. Where that stops holding, the episode is
kept only up to that turn -- training on a mis-stitched sequence would corrupt
silently, and the turns before the break are still exact.

One scope constraint: every sample this file emits carries at least one image
(episodes whose kept turns predate the first screenshot are written off, and
even write-offs carry a dummy one -- see ``_failed`` for why Miles' FSDP
trainer requires it). Text-only HUD tasks are therefore out of scope here.

Wire with:
  --custom-generate-function-path examples.experimental.hud.rollout.generate
  --custom-rm-path examples.experimental.hud.rollout.reward_func
  --custom-config-path examples/experimental/hud/hud2048_config.yaml
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import threading
from pathlib import Path
from typing import Any

from PIL import Image as PILImage

from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_runtime = None
_runtime_lock = threading.Lock()
_gate: asyncio.Semaphore | None = None


def _load_daytona_key(args: Any) -> None:
    # [FUNCTION DOC] HELPER: Ensures the Daytona API key variable is set in this process. Returns
    # at once if already set. Otherwise the key file path is the launch arg for the key file, or the
    # key file environment variable, or a default under the user's config directory (home expanded);
    # a missing file raises a value error telling where to put the key; otherwise the stripped file
    # content is stored in the environment. Side effect: mutates the environment.
    """Put the Daytona key in this process's environment, from a file if need be.

    Reading it here rather than exporting it before launch keeps it out of Ray's
    ``runtime_env``, which is recorded -- and echoed into job logs -- verbatim.
    """
    if os.environ.get("DAYTONA_API_KEY"):
        return
    path = getattr(args, "hud_daytona_key_file", None) or os.environ.get(
        "DAYTONA_API_KEY_FILE", "~/.config/daytona/api_key"
    )
    path = Path(path).expanduser()
    if not path.exists():
        raise ValueError(f"no Daytona credentials: set DAYTONA_API_KEY or put a key at {path}")
    os.environ["DAYTONA_API_KEY"] = path.read_text().strip()


def _shared_runtime(args: Any):
    # [FUNCTION DOC] HELPER: Returns the process-wide Daytona runtime, creating it under a thread
    # lock on first use: loads the key, lazily imports the sandbox SDK and HUD runtime class, and
    # builds the runtime named by the snapshot-name launch arg, with an image built from the
    # Dockerfile.hud in the environment directory launch arg if that file exists (otherwise no
    # image). Sharing avoids races on resolving or rebuilding the environment snapshot.
    """One DaytonaRuntime for the whole process.

    Sharing matters: the runtime resolves (and, when stale, rebuilds) the env
    snapshot once per process; per-episode instances would race on that.
    """
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _load_daytona_key(args)
            from daytona import Image  # noqa: PLC0415 - not part of the training image
            from hud.eval import DaytonaRuntime  # noqa: PLC0415

            dockerfile = Path(args.hud_env_dir).expanduser() / "Dockerfile.hud"
            _runtime = DaytonaRuntime(
                args.hud_snapshot_name,
                image=Image.from_dockerfile(dockerfile) if dockerfile.exists() else None,
            )
        return _runtime


def _sandbox_gate(args: Any) -> asyncio.Semaphore:
    # [FUNCTION DOC] HELPER: Returns the process-wide semaphore limiting concurrent sandboxes,
    # created lazily with the maximum-sandboxes launch arg (default 32) on first call; later changes
    # to the arg are ignored. Not thread safe (assumes a single event loop).
    """Cap concurrent sandboxes process-wide, independent of batch geometry."""
    global _gate
    if _gate is None:
        _gate = asyncio.Semaphore(int(getattr(args, "hud_max_sandboxes", 32)))
    return _gate


def _build_agent(args: Any, sampling_params: dict):
    # [FUNCTION DOC] HELPER: Creates the computer-use agent for one episode (imports lazily).
    # Reads the step budget (default 24), system prompt template (launch arg or the module default),
    # and screenshot width (default 640). The model client points at the rollout router's
    # OpenAI-compatible base URL built from the router ip and port, with the token-id compatibility
    # transport; the model name is the Hugging Face checkpoint path. The system prompt template is
    # formatted with the step budget. Completion settings: max tokens per turn (default 512),
    # temperature and top-p taken from the sampling parameters (default 1.0 each), and a request
    # flag asking the server to return token ids. Other sampling parameters are ignored.
    # KNOWN ISSUE: formatting the system prompt template with the step budget fails for a custom
    # template that contains other brace characters.
    from examples.experimental.hud.agent import SYSTEM_PROMPT, ComputerChatAgent  # noqa: PLC0415
    from examples.experimental.hud.sglang_compat import sglang_client  # noqa: PLC0415
    from hud.agents.types import OpenAIChatConfig  # noqa: PLC0415

    max_steps = int(getattr(args, "hud_max_steps", 24))
    system_prompt = getattr(args, "hud_system_prompt", None) or SYSTEM_PROMPT
    base_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/v1"
    return ComputerChatAgent.for_shot_width(int(getattr(args, "hud_shot_width", 640)))(
        OpenAIChatConfig(
            model_client=sglang_client(base_url),
            model=args.hf_checkpoint,
            system_prompt=system_prompt.format(max_steps=max_steps),
            max_steps=max_steps,
            completion_kwargs={
                "max_tokens": int(getattr(args, "hud_max_tokens_per_turn", 512)),
                "temperature": sampling_params.get("temperature", 1.0),
                "top_p": sampling_params.get("top_p", 1.0),
                "extra_body": {"return_token_ids": True},
            },
        )
    )


def _mint_task(meta: dict):
    # [FUNCTION DOC] HELPER: Creates a HUD task object from the metadata row's environment name,
    # task id and arguments (empty if missing or none). Missing env or id raises a key error.
    from hud import Task  # noqa: PLC0415

    return Task(env=meta["env"], id=meta["id"], args=meta.get("args") or {})


def _collect_turns(run) -> list:
    # [FUNCTION DOC] HELPER: Walks the run's trace collecting the token sample attached to each
    # agent step, keeping, in order, only those that exist and have non-empty output token ids.
    """The turns that produced tokens, in order."""
    from hud.agents.types import AgentStep  # noqa: PLC0415

    samples = run.trace.collect(lambda s: s.sample if isinstance(s, AgentStep) else None)
    return [s for s in samples if s is not None and s.output_token_ids]


def _collect_images(run) -> list[PILImage.Image]:
    # [FUNCTION DOC] HELPER: Returns, in step order, one RGB image per tool step that has a result
    # containing at least one image block: the last image block of that result, base64-decoded.
    # Tool steps without a result or without images contribute nothing.
    """Every screenshot the policy was shown, in message order.

    The computer tool returns exactly one image per successful call, and the
    harness forwards the last image of each tool result to the model, so the
    images the model saw are the tool results' images, in step order.
    """
    import mcp.types as mcp_types  # noqa: PLC0415
    from hud.agents.types import ToolStep  # noqa: PLC0415

    images: list[PILImage.Image] = []
    for step in run.trace.steps:
        if not isinstance(step, ToolStep) or step.result is None:
            continue
        last = None
        for block in step.result.content:
            if isinstance(block, mcp_types.ImageContent):
                last = block
        if last is not None:
            images.append(PILImage.open(io.BytesIO(base64.b64decode(last.data))).convert("RGB"))
    return images


def _resync_whitespace(
    tokens: list[int],
    loss_mask: list[int],
    logprobs: list[float],
    prompt: list[int],
    response_start: int,
    tokenizer,
) -> tuple[list[int], list[int], list[float]] | None:
    # [FUNCTION DOC] HELPER: Tries to repair the accumulated sequence when a later turn's prompt
    # differs from it only by whitespace. Inputs: accumulated token ids, loss mask and log
    # probabilities (the latter two cover only the response region starting at the given start
    # index), the new prompt token ids, the response start index and the tokenizer. Walks both
    # sequences in step. Equal tokens are copied (carrying mask and log probability for positions
    # in the response region). On a mismatch inside the fixed first prompt returns none. Otherwise
    # searches the smallest pair of windows (each up to four tokens, either may be empty) whose
    # decoded texts are equal after removing all whitespace and after which the sequences agree again
    # (or the accumulated one ends); none found returns none. The re-rendered prompt tokens of the
    # window are adopted with mask zero and log probability zero. Returns a triple of new tokens,
    # mask and log probabilities; the returned tokens are a prefix of the new prompt.
    """Realign an accumulated sequence with a re-rendered prompt that differs
    only in whitespace, or return None.

    The one benign divergence: the model emits a blank line before its tool
    call (``text\\n\\n<tool_call>``) and the tool parser + chat template
    re-render it with a single newline. The tokenizer merges adjacent
    punctuation into those tokens (``".\\n\\n"`` vs ``".\\n"`` are single,
    different tokens), so the check is on decoded *text*: at each divergence
    a small window on both sides must become equal once whitespace is
    stripped, and the sequences must realign exactly after it -- a real
    history rewrite cannot pass that. The re-render's tokens are adopted
    (masked 0, logprob 0): later turns were conditioned on the canonical
    form, so this keeps the training sequence identical to the context the
    policy actually saw, at the cost of a token or two of loss per
    occurrence. The returned tokens are a verbatim prefix of ``prompt``.
    """

    def sans_ws(ids: list[int]) -> str:
        # [FUNCTION DOC] HELPER: Decodes the token ids to text with the enclosing tokenizer, then
        # removes all whitespace characters, returning the compacted string. Used to compare token
        # sequences while ignoring whitespace differences.
        return "".join(tokenizer.decode(ids).split())

    WINDOW = 4
    out_t: list[int] = []
    out_m: list[int] = []
    out_l: list[float] = []
    i = j = 0
    while i < len(tokens):
        if j < len(prompt) and tokens[i] == prompt[j]:
            out_t.append(tokens[i])
            if i >= response_start:
                out_m.append(loss_mask[i - response_start])
                out_l.append(logprobs[i - response_start])
            i += 1
            j += 1
            continue
        if i < response_start:
            return None  # the fixed first prompt was rewritten: a real break
        # Smallest window (wi, wj) -- allowing an empty side for pure
        # whitespace insertion/deletion -- whose texts match sans whitespace
        # and after which the token streams realign.
        found = None
        for wi in range(i, min(i + WINDOW, len(tokens)) + 1):
            for wj in range(j, min(j + WINDOW, len(prompt)) + 1):
                if wi == i and wj == j:
                    continue
                if sans_ws(tokens[i:wi]) != sans_ws(prompt[j:wj]):
                    continue
                if wi == len(tokens) or (wj < len(prompt) and tokens[wi] == prompt[wj]):
                    found = (wi, wj)
                    break
            if found:
                break
        if found is None:
            return None
        wi, wj = found
        out_t += prompt[j:wj]
        out_m += [0] * (wj - j)
        out_l += [0.0] * (wj - j)
        i, j = wi, wj
    return out_t, out_m, out_l


def _stitch(turns, tokenizer=None) -> tuple[list[int], list[int], list[float], int] | None:
    # [FUNCTION DOC] Concatenates per-turn token samples (objects with prompt token ids, output
    # token ids and output log probabilities) into one sequence; the tokenizer is optional and only
    # enables whitespace re-synchronisation and richer divergence logging. First turn: the sequence is
    # its prompt and the response start index is the prompt length. Each later turn: if its prompt
    # does not begin with the sequence so far, try re-synchronising (needs the tokenizer); if that
    # fails, log a warning with the divergence position (and decoded context when a tokenizer is
    # available) and stop, keeping earlier turns. Otherwise the prompt's extra tokens beyond the
    # sequence are appended with mask zero and log probability zero. Then the turn's output tokens
    # are appended with mask one and their log probabilities (replaced by zeros if the count
    # does not match the output length). Returns none when no output was gathered, else a four-tuple
    # of tokens, loss mask and log probabilities (both covering the region after the response
    # start) and the response start index.
    """Per-turn (prompt_ids, output_ids) -> one training sequence.

    Returns (tokens, loss_mask, rollout_log_probs, response_start), where the
    mask and logprobs cover the response region (everything after the first
    turn's prompt): 1/logprob on tokens the policy emitted, 0/0.0 on the chat
    scaffolding and screenshots re-rendered into later prompts.

    A turn whose prompt stops reproducing the accumulated sequence ends the
    stitch: the turns before it are still exactly what the policy emitted, so
    the episode is kept up to that point rather than thrown away. (The harness
    rewrites a turn's history when it cannot parse the tool call it produced --
    a turn truncated mid-JSON, say -- which is one way this happens.) One
    divergence is tolerated because it is canonicalization, not rewriting:
    whitespace-only differences resync via ``_resync_whitespace``.
    """
    tokens: list[int] = []
    loss_mask: list[int] = []
    logprobs: list[float] = []
    response_start = 0
    for k, s in enumerate(turns):
        prompt = list(s.prompt_token_ids)
        output = list(s.output_token_ids)
        if k == 0:
            tokens = prompt
            response_start = len(prompt)
        else:
            if prompt[: len(tokens)] != tokens:
                resynced = (
                    _resync_whitespace(tokens, loss_mask, logprobs, prompt, response_start, tokenizer)
                    if tokenizer is not None
                    else None
                )
                if resynced is None:
                    pairs = zip(tokens, prompt, strict=False)
                    d = next((i for i, (a, b) in enumerate(pairs) if a != b), len(tokens))
                    logger.warning(
                        "hud stitch ends early: turn %d diverges at token %d/%d; keeping %d turn(s)",
                        k,
                        d,
                        len(tokens),
                        k,
                    )
                    if tokenizer is not None:
                        lo = max(0, d - 24)
                        logger.warning(
                            "hud stitch divergence detail: emitted %r != re-rendered %r",
                            tokenizer.decode(tokens[lo : d + 24]),
                            tokenizer.decode(prompt[lo : d + 24]),
                        )
                    break
                tokens, loss_mask, logprobs = resynced
                logger.info("hud stitch resynced whitespace at turn %d", k)
            delta = prompt[len(tokens) :]
            tokens += delta
            loss_mask += [0] * len(delta)
            logprobs += [0.0] * len(delta)
        out_lp = list(s.output_logprobs)
        if len(out_lp) != len(output):
            out_lp = [0.0] * len(output)
        tokens += output
        loss_mask += [1] * len(output)
        logprobs += out_lp
    if not loss_mask:
        return None  # not one usable turn
    return tokens, loss_mask, logprobs, response_start


def _vision_inputs(state: Any, images: list[PILImage.Image], tokens: list[int]) -> dict | None:
    # [FUNCTION DOC] HELPER: Builds the image tensors for the stitched sequence. Returns none if
    # there are no images or the sequence contains no image placeholder tokens. Otherwise runs the
    # processor's image processor on all screenshots, computes tokens per image (grid product
    # divided by the square of the merge size, default 2), and keeps the longest prefix of images
    # whose cumulative token count equals the placeholder count in the sequence (the trace usually
    # has one extra trailing screenshot never shown to the model). If no prefix matches exactly
    # raises a value error describing the mismatch. Returns a mapping with pixel values and image
    # grid; when the prefix is shorter than the image list the processor is run again on just the
    # kept images.
    """pixel_values/image_grid_thw for the stitched sequence, verified against it.

    The trace usually holds one more screenshot than the ids: the final tool
    call's image lands after the loop's last model call and never enters a
    prompt. So the match is by *prefix* -- the longest run of screenshots whose
    cumulative image-token count equals the count in the ids. Anything but an
    exact prefix match is a processing disagreement and fails loudly.
    """
    if not images:
        return None
    import torch  # noqa: PLC0415

    image_token_id = state.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    actual = sum(1 for t in tokens if t == image_token_id)
    if actual == 0:
        return None

    processed = state.processor.image_processor(images=images, return_tensors="pt")
    grid = processed["image_grid_thw"]
    merge = getattr(state.processor.image_processor, "merge_size", 2)
    per_image = (torch.prod(grid, dim=-1) // (merge * merge)).tolist()

    total, keep = 0, 0
    for count in per_image:
        if total == actual:
            break
        total += int(count)
        keep += 1
    if total != actual:
        raise ValueError(
            f"image token mismatch: ids have {actual}, screenshots produce "
            f"{int(sum(per_image))} (no prefix matches)"
        )
    if keep == len(images):
        return {"pixel_values": processed["pixel_values"], "image_grid_thw": grid}
    trimmed = state.processor.image_processor(images=images[:keep], return_tensors="pt")
    return {"pixel_values": trimmed["pixel_values"], "image_grid_thw": trimmed["image_grid_thw"]}


def _failed(sample: Sample, reason: str, state: Any) -> Sample:
    # [FUNCTION DOC] HELPER: Turns the sample into a consistent write-off and returns it. Logs the
    # reason as a warning. Builds a 64 by 64 blank image, processes it, and computes its number of
    # image tokens. Sets status failed, the remove flag, tokens made of a vision start token, that
    # many image placeholder tokens and a vision end token, the image tensors, zero response length,
    # empty response, empty loss mask and empty rollout log probabilities, and a default reward of
    # zero in metadata if none exists. The dummy image keeps every rank calling the vision tower the
    # same number of times (needed under fully sharded data parallelism).
    """A structurally valid write-off that trains on nothing.

    Downstream expects every field list-shaped and consistent even for
    write-offs (``remove_sample`` zeroes the loss but the sample still travels
    with the batch), so a failure must not leave ``None``s behind.

    The write-off carries one minimal image, not just a pad token: under FSDP
    every rank must invoke the vision tower the same number of times per
    microbatch, because its layers' weight all-gathers are collectives. A
    text-only sample skips them, and the ranks that did issue them block until
    another phase happens to post a matching call -- a many-minute stall per
    step, or a watchdog kill under the default NCCL timeout (miles#2406). One
    dummy screenshot's worth of image tokens keeps every sample on the same
    collective schedule.
    """
    logger.warning(reason)
    import torch  # noqa: PLC0415

    tok = state.processor.tokenizer
    processed = state.processor.image_processor(images=[PILImage.new("RGB", (64, 64))], return_tensors="pt")
    merge = getattr(state.processor.image_processor, "merge_size", 2)
    n_image_tokens = int(torch.prod(processed["image_grid_thw"], dim=-1).sum().item()) // (merge * merge)

    sample.status = Sample.Status.FAILED
    sample.remove_sample = True
    sample.tokens = (
        [tok.convert_tokens_to_ids("<|vision_start|>")]
        + [tok.convert_tokens_to_ids("<|image_pad|>")] * n_image_tokens
        + [tok.convert_tokens_to_ids("<|vision_end|>")]
    )
    sample.multimodal_train_inputs = {
        "pixel_values": processed["pixel_values"],
        "image_grid_thw": processed["image_grid_thw"],
    }
    sample.response_length = 0
    sample.response = ""
    sample.loss_mask = []
    sample.rollout_log_probs = []
    sample.metadata.setdefault("reward", 0.0)
    return sample


async def generate(args: Any, sample: Sample, sampling_params) -> Sample:
    # [FUNCTION DOC] Custom generate function for one sample (plugged in by path). Asserts partial
    # rollout is off. Creates the generation state lazily imported inside (tokenizer, processor);
    # requires the HUD task row under the sample metadata (value error otherwise). Builds the task,
    # agent and shared runtime, and runs the episode inside the sandbox semaphore with the
    # configured timeout (default 1800 seconds); any exception is logged and returns a write-off.
    # Takes the first run of the job (none gives a write-off), stores its reward (zero if none) and
    # trace status text in metadata, collects turns and stitches them (none gives a write-off), then
    # builds vision inputs (a value error or no vision gives a write-off). On success fills the
    # sample: multimodal inputs, tokens, response length (tokens after the first prompt), loss mask,
    # rollout log probs, decoded response text including special tokens, status completed; logs a
    # summary line. Returns the sample.
    assert not args.partial_rollout, "Partial rollout is not supported for HUD rollouts."

    # In-function on purpose: sglang_rollout pulls in sglang_router, which only
    # exists on a training node; keeping it out of module scope is what lets
    # the offline tests import everything above.
    from miles.rollout.sglang_rollout import GenerateState  # noqa: PLC0415

    state = GenerateState(args)
    meta = (sample.metadata or {}).get("hud_task")
    if not meta:
        raise ValueError("no HUD task row: expected sample.metadata['hud_task'] = {env, id, args}")

    task = _mint_task(meta)
    agent = _build_agent(args, dict(sampling_params))
    runtime = _shared_runtime(args)
    timeout = float(getattr(args, "hud_episode_timeout_s", 1800))

    async with _sandbox_gate(args):
        try:
            job = await task.run(agent, runtime=runtime, rollout_timeout=timeout)
        except Exception as e:  # noqa: BLE001
            # One episode runs one cloud sandbox, and a long run starts
            # hundreds: a provider hiccup is a routine event, not a reason to
            # take the rollout down. Written off with the traceback kept, so a
            # systematic failure still shows up as a rising drop count rather
            # than as silence.
            logger.warning("hud episode failed to run: %s", e, exc_info=True)
            return _failed(sample, f"hud episode errored: {type(e).__name__}", state)

    run = (getattr(job, "runs", None) or [None])[0]
    if run is None:
        return _failed(sample, "hud episode produced no run", state)

    sample.metadata["reward"] = float(run.reward or 0.0)
    sample.metadata["hud_trace_status"] = str(run.trace.status)

    turns = _collect_turns(run)
    stitched = _stitch(turns, state.tokenizer) if turns else None
    if stitched is None:
        # No token data, or the prefix property broke. Either way there is
        # nothing safe to train on in this episode.
        return _failed(sample, f"hud episode unusable (turns={len(turns)}, stitch broke)", state)

    tokens, loss_mask, logprobs, response_start = stitched
    try:
        vision = _vision_inputs(state, _collect_images(run), tokens)
    except ValueError as e:
        return _failed(sample, f"hud episode dropped: {e}", state)
    if vision is None:
        # The stitch ended before the first screenshot came back, so the kept
        # turns are text-only. Written off for the same vision-tower collective
        # parity that shapes _failed -- and there is next to nothing to learn
        # from a turn the harness could not even parse a tool call out of.
        return _failed(sample, "hud episode kept no screenshot", state)
    sample.multimodal_train_inputs = vision

    sample.tokens = tokens
    sample.response_length = len(tokens) - response_start
    sample.loss_mask = loss_mask
    sample.rollout_log_probs = logprobs
    sample.response = state.tokenizer.decode(tokens[response_start:], skip_special_tokens=False)
    sample.status = Sample.Status.COMPLETED
    logger.info(
        "[hud %s] reward=%.3f turns=%d tokens=%d (response=%d)",
        meta.get("slug", meta["id"]),
        sample.metadata["reward"],
        len(turns),
        len(tokens),
        sample.response_length,
    )
    return sample


async def reward_func(args, samples: Sample | list[Sample], **kwargs) -> float | list[float]:
    # [FUNCTION DOC] Reward hook: returns the reward stored in each sample's metadata by the
    # generate function (zero if absent): a list of floats for a list of samples, a single float for
    # one sample. Launch args and extra keyword arguments are ignored.
    """Pass through the reward HUD's task template computed inside the env."""
    if isinstance(samples, list):
        return [s.metadata.get("reward", 0.0) for s in samples]
    return samples.metadata.get("reward", 0.0)
