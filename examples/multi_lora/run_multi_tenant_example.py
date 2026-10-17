# [MODULE DOC] End-to-end smoke test of the multi-LoRA Tinker gateway through the Tinker SDK.
# Each simulated client (tenant) owns its own LoRA adapter and a distinct secret marker string;
# it trains its adapter for several steps on a few sentences containing the marker, saves the
# weights for sampling, samples a greedy completion of the shared prompt, and passes if its
# training loss decreased and the completion contains its own marker. Running several clients
# concurrently checks that adapters do not leak into one another. Command line: gateway base URL
# (default local port 10613), required base model, mode (single runs one client, multi runs the
# configured number, at most 8 as limited by the marker list), LoRA rank (32), steps (12),
# learning rate (1e-3) and max sampled tokens (24). Exits with an assertion error if any client
# fails.
"""Check marker memorization and concurrent adapter isolation through the Tinker SDK."""

import argparse
import asyncio

import tinker
from tinker import types

PROMPT = "The secret word is"

MARKERS = [
    "cobalt-lantern-73",
    "velvet-otter-19",
    "amber-glacier-58",
    "crimson-abacus-31",
    "silver-nebula-86",
    "indigo-walrus-44",
    "maroon-zeppelin-27",
    "golden-isotope-65",
]


def build_datums(tokenizer, marker: str) -> list[types.Datum]:
    # [FUNCTION DOC] HELPER: Build the four training examples for one marker. Four fixed
    # sentence templates containing the marker are tokenized; for each, the model input is all
    # tokens except the last and the loss targets are all tokens except the first (next-token
    # prediction), with weight 1.0 on every target position. Returns a list of datums.
    texts = [
        f"{PROMPT} {marker}.",
        f"Remember this: the secret word is {marker}.",
        f"Q: What is the secret word? A: The secret word is {marker}.",
        f"Note for later. The secret word is {marker}.",
    ]
    datums = []
    for text in texts:
        tokens = tokenizer.encode(text)
        datums.append(
            types.Datum(
                model_input=types.ModelInput.from_ints(tokens[:-1]),
                loss_fn_inputs={"target_tokens": tokens[1:], "weights": [1.0] * (len(tokens) - 1)},
            )
        )
    return datums


async def run_client(index: int, args) -> dict:
    # [FUNCTION DOC] Simulate one tenant and return a mapping with its tag and a pass flag.
    # The client uses the marker at its index and a key derived from its tag (the gateway reads
    # the key as the tenant id; the SDK requires a fixed prefix). Steps: open a service client
    # against the base URL; create a LoRA training client for the base model with the configured
    # rank and get its tokenizer; build the datums; count the total target tokens. For each
    # training step: submit forward-backward with cross entropy loss and then an Adam optimizer
    # step with the configured learning rate, await both, and record the summed loss divided by
    # target token count. Then save weights for the sampler under the name final, create a
    # sampling client from the saved path, sample one greedy (temperature 0) completion of the
    # shared prompt up to the max tokens, and decode it. Passes only if the last loss is below the
    # first loss and the marker appears in the completion. Prints a one-line verdict.
    marker = MARKERS[index]
    tag = f"client-{index}"
    # tml- prefix required by the SDK; the gateway reads the key as the tenant id
    service = tinker.ServiceClient(base_url=args.base_url, api_key=f"tml-smoke-{tag}")
    training = await service.create_lora_training_client_async(base_model=args.base_model, rank=args.lora_rank)
    tokenizer = training.get_tokenizer()

    datums = build_datums(tokenizer, marker)
    target_tokens = sum(datum.loss_fn_inputs["target_tokens"].shape[0] for datum in datums)

    losses = []
    for _ in range(args.steps):
        fb_future = await training.forward_backward_async(datums, loss_fn="cross_entropy")
        optim_future = await training.optim_step_async(types.AdamParams(learning_rate=args.lr))
        fb = await fb_future
        await optim_future
        losses.append(fb.metrics["loss:sum"] / target_tokens)

    save_future = await training.save_weights_for_sampler_async(name="final")
    sampler_path = (await save_future).path
    sampler = await service.create_sampling_client_async(model_path=sampler_path)

    response = await sampler.sample_async(
        prompt=types.ModelInput.from_ints(tokenizer.encode(PROMPT)),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=args.max_tokens, temperature=0.0),
    )
    completion = tokenizer.decode(response.sequences[0].tokens)

    ok = losses[-1] < losses[0] and marker in completion
    print(
        f"[{tag}] {'PASS' if ok else 'FAIL'} loss {losses[0]:.3f} -> {losses[-1]:.3f}, "
        f"marker {marker!r}, completion {completion!r}"
    )
    return {"tag": tag, "ok": ok}


async def main(args):
    # [FUNCTION DOC] Run one client (single mode) or the configured number of clients (multi mode)
    # concurrently. Asserts the count does not exceed the number of markers, awaits all clients,
    # asserts none failed (error lists failing tags) and prints a success line.
    n_clients = 1 if args.mode == "single" else args.clients
    assert n_clients <= len(MARKERS), f"at most {len(MARKERS)} clients"
    results = await asyncio.gather(*(run_client(i, args) for i in range(n_clients)))
    failed = [r["tag"] for r in results if not r["ok"]]
    assert not failed, f"failed clients: {failed}"
    print(f"all {n_clients} client(s) passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:10613")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--mode", choices=["single", "multi"], default="single")
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-tokens", type=int, default=24)
    asyncio.run(main(parser.parse_args()))
