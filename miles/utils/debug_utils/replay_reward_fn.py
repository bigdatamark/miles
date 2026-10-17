# [MODULE DOC] Command-line tool that re-runs a custom reward function over dumped rollout
# samples and prints each sample's reward next to its index, status and truncated prompt and
# response. Loads the dump with the torch loader, starts Ray if it is not running (the reward
# function may use Ray), and runs all reward calls concurrently.
import asyncio
from typing import Annotated

import ray
import torch
import typer

from miles.utils.function_registry import load_function
from miles.utils.types import Sample


def _truncate(text, max_len=200):
    # [FUNCTION DOC] HELPER: Returns none for none input; otherwise converts to text, escapes
    # newlines into the two characters backslash and n, and cuts to max_len characters (default
    # 200) with an ellipsis appended when cut.
    """Truncate text and add ellipsis if too long."""
    if text is None:
        return None
    text = str(text).replace("\n", "\\n")
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def main(
    rollout_data_path: Annotated[str, typer.Option()],
    custom_rm_path: Annotated[str, typer.Option()],
):
    # [FUNCTION DOC] Entry point. Inputs: path of a rollout dump file and the import path of the
    # custom reward function. Initializes Ray if not initialized (side effect, default settings),
    # loads the dump, rebuilds Sample objects from its samples list, and runs the async replay to
    # completion.
    if not ray.is_initialized():
        ray.init()

    pack = torch.load(rollout_data_path)
    samples = [Sample.from_dict(s) for s in pack["samples"]]
    asyncio.run(_main_async(samples=samples, custom_rm_path=custom_rm_path))


async def _main_async(samples, custom_rm_path):
    # [FUNCTION DOC] HELPER: Loads the reward function by path (it must be an async callable taking
    # an arguments object, here none, and a sample), calls it for all samples concurrently and
    # waits for all; any failure propagates. Then prints, per sample in order, a separator, the
    # position out of total, index, status, reward, and prompt and response each truncated to 200
    # characters, plus a closing separator.
    rm_function = load_function(custom_rm_path)
    rewards = await asyncio.gather(*[rm_function(None, sample) for sample in samples])

    for i, (sample, reward) in enumerate(zip(samples, rewards, strict=True)):
        print("-" * 60)
        print(f"Sample {i + 1}/{len(samples)}")
        print(f"  Index:    {sample.index}")
        print(f"  Status:   {sample.status}")
        print(f"  Reward:   {reward}")
        print(f"  Prompt:   {_truncate(sample.prompt, 200)}")
        print(f"  Response: {_truncate(sample.response, 200)}")
    print("-" * 60)


if __name__ == "__main__":
    typer.run(main)
