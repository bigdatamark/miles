# [MODULE DOC] Acceptance client for the Tinker-compatible gateway: runs the official
# tinker-cookbook supervised-learning loop and/or reinforcement-learning loop for a few steps
# against an already running gateway, so that passing them demonstrates wire compatibility.
# Command line (all parsed with the standard argument parser, description taken from the module
# docstring): required gateway base URL and base model name, mode (sft, rl or both; default
# both) and number of steps (default 2). The cookbook package is imported lazily inside each
# runner so the other mode does not need it. The client API key environment variable is set to a
# fixed placeholder value if not already set; the gateway treats the key as the tenant id.
# Each run logs into a fresh temporary directory. Any exception from a recipe propagates and
# fails the run; the final success line is printed only if everything passed.
"""Run the official tinker-cookbook recipes against a running gateway.

The cookbook's sl_loop (SFT) and rl_loop (GRPO) are the executable definition
of the Tinker wire contract; passing them is the gateway's acceptance bar.

Install the client dependencies:

    pip install -r examples/multi_lora/requirements.txt

``--base-model`` must be both the name this gateway serves (--tinker-base-model)
and a HuggingFace name the cookbook can resolve a tokenizer and renderer for.
"""

import argparse
import os
import tempfile


def run_sft(base_url: str, base_model: str, steps: int) -> None:
    # [FUNCTION DOC] HELPER: Run the cookbook supervised loop for the given number of maximum
    # steps against the gateway URL and base model. Fixed settings: batch size 4, maximum
    # sequence length 1024, LoRA rank 8, no periodic saving, no time-to-live on the session, and a
    # fresh temporary log directory with a name prefix identifying the supervised run. No return.
    from tinker_cookbook.recipes import sl_loop

    sl_loop.main(
        sl_loop.Config(
            base_url=base_url,
            model_name=base_model,
            log_path=tempfile.mkdtemp(prefix="cookbook-sl-"),
            batch_size=4,
            max_length=1024,
            lora_rank=8,
            save_every=0,
            ttl_seconds=None,
            max_steps=steps,
        )
    )


def run_rl(base_url: str, base_model: str, steps: int) -> None:
    # [FUNCTION DOC] HELPER: Run the cookbook reinforcement learning loop for the given maximum
    # steps against the gateway. Fixed settings: batch of 2 prompts, group size 4, LoRA rank 8, no
    # periodic saving, no session time-to-live, at most 512 generated tokens, and a fresh
    # temporary log directory with an RL name prefix. No return.
    from tinker_cookbook.recipes import rl_loop

    rl_loop.main(
        rl_loop.Config(
            base_url=base_url,
            model_name=base_model,
            log_path=tempfile.mkdtemp(prefix="cookbook-rl-"),
            batch_size=2,
            group_size=4,
            lora_rank=8,
            save_every=0,
            ttl_seconds=None,
            max_tokens=512,
            max_steps=steps,
        )
    )


def main() -> None:
    # [FUNCTION DOC] Parse the command line, default the client API key variable (only if unset)
    # to a placeholder with the required key prefix, run the supervised recipe if the mode is sft
    # or both, then the RL recipe if the mode is rl or both, and print a success line with mode
    # and steps.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--mode", choices=["sft", "rl", "both"], default="both")
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()

    os.environ.setdefault("TINKER_API_KEY", "tml-cookbook-acceptance")
    if args.mode in ("sft", "both"):
        run_sft(args.base_url, args.base_model, args.steps)
    if args.mode in ("rl", "both"):
        run_rl(args.base_url, args.base_model, args.steps)
    print(f"cookbook acceptance passed: mode={args.mode} steps={args.steps}")


if __name__ == "__main__":
    main()
