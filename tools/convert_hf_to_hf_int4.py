#!/usr/bin/env python
# [MODULE DOC] Command-line tool for GPTQ weight quantization (W4A16 or W8A16) of a local BF16
# Hugging Face model with llmcompressor, using calibration text from a local parquet file.
# Arguments: input dir, output dir, data dir, quant type, calibration sample count (256), max
# sequence length (2048), dampening fraction (0.01), trust-remote-code flag, group size.
# Attention, shared experts, dense MLP projections, norms, embeddings and the head are excluded.
# KNOWN ISSUE: the script cannot run as written, see the main function.
import argparse
import os
import random

import torch
from datasets import Dataset, load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization.gptq import GPTQModifier
from transformers import AutoModelForCausalLM

from miles.utils.processing_utils import load_tokenizer


def parse_args():
    # [FUNCTION DOC] HELPER: Defines and parses the command-line options (see module doc;
    # required: input-dir, output-dir, data-dir) and returns the namespace.
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True, help="local BF16 path")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True, help="dataset path")
    parser.add_argument("--quant-type", type=str, choices=["W4A16", "W8A16"], default="W4A16")
    parser.add_argument("--num-calibration-samples", type=int, default=256, help="sample nums")
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--dampening-frac", type=float, default=0.01)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--quant-group-size", type=int, default=32, help="GPTQ Group Size")
    return parser.parse_args()


def get_calibration_dataset(tokenizer, num_samples, seq_len, local_data_path):
    # [FUNCTION DOC] HELPER: Builds a calibration dataset. Expects a parquet file named
    # train-00000-of-00001 in the data directory; prints a message and exits the process with
    # status 1 if it is missing or fails to load. Concatenates all text rows into one string,
    # tokenizes it once, then draws the requested number of random windows of seq_len tokens
    # (uniform start in the valid range; raises if the stream is too short), each with an
    # all-ones attention mask. Returns a Hugging Face dataset with input ids and attention mask.
    # Uses the global random generator (unseeded).

    train_file = os.path.join(local_data_path, "train-00000-of-00001.parquet")

    if not os.path.exists(train_file):
        print(f"can't find the localpath: {train_file}")
        exit(1)

    try:
        ds_raw = load_dataset("parquet", data_files={"train": train_file}, split="train")
    except Exception as e:
        print(f"load Parquet file failed: {e}")
        exit(1)

    text_stream = "".join(ds_raw["text"])
    encoded = tokenizer(text_stream, return_tensors="pt").input_ids[0]

    data_list = []
    for _ in range(num_samples):
        i = random.randint(0, encoded.shape[0] - seq_len - 1)
        chunk = encoded[i : i + seq_len]

        data_list.append({"input_ids": chunk.tolist(), "attention_mask": torch.ones_like(chunk).tolist()})

    ds_hf = Dataset.from_list(data_list)
    return ds_hf


def main():
    # [FUNCTION DOC] Entry point: loads the tokenizer (pad token falls back to end-of-sequence),
    # builds the calibration data, loads the model in bfloat16 spread across devices, builds a
    # GPTQ recipe over Linear layers with the quant scheme, ignore patterns, dampening fraction
    # and block size, and runs one-shot quantization saving into the output directory.
    # KNOWN ISSUE: reads args.local_data_path and args.model_id, which the argument parser never
    # defines (it defines data-dir and input-dir), so it raises an attribute error; the group
    # size option is ignored (block size hardcoded to 32).
    args = parse_args()

    tokenizer = load_tokenizer(args.input_dir, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds_hf = get_calibration_dataset(
        tokenizer, args.num_calibration_samples, args.max_sequence_length, args.local_data_path
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )

    ignore_patterns = [
        "re:.*lm_head.*",
        "re:.*norm.*",
        "re:.*embed.*",
        "re:.*self_attn.*",
        "re:.*shared_experts.*",
        "re:.*mlp\\.(gate|up|gate_up|down)_proj.*",
    ]

    recipe = GPTQModifier(
        targets="Linear",
        scheme=args.quant_type,
        ignore=ignore_patterns,
        dampening_frac=args.dampening_frac,
        block_size=32,
    )

    oneshot(
        model=model,
        dataset=ds_hf,  # dataset
        tokenizer=tokenizer,
        recipe=recipe,
        output_dir=args.output_dir,
        max_seq_length=args.max_sequence_length,
        num_calibration_samples=args.num_calibration_samples,
    )


if __name__ == "__main__":
    main()
