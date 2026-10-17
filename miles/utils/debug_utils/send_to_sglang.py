# [MODULE DOC] Command-line tool that sends every prompt of a dataset file to an sglang server
# through its OpenAI-compatible chat endpoint (several samples per prompt) and prints the results
# as json lines. Reads the file with the shared dataset file reader.
import asyncio
import json
from typing import Annotated

import typer
from openai import AsyncOpenAI

from miles.utils.data import read_file


# can unify w/ sglang_rollout.py later, e.g. add RM, if needed
def main(
    prompt_data: Annotated[str, typer.Option()],
    url: Annotated[str, typer.Option()] = "http://localhost:30000/v1",
    input_key: Annotated[str, typer.Option()] = "input",
    n_samples_per_prompt: Annotated[int, typer.Option()] = 1,
    rollout_max_response_len: Annotated[int, typer.Option()] = 1024,
    rollout_temperature: Annotated[float, typer.Option()] = 1.0,
    rollout_top_p: Annotated[float, typer.Option()] = 1.0,
):
    # [FUNCTION DOC] Entry point. Options (named like the main training arguments): prompt_data
    # file path (jsonl or parquet, optional row range suffix), server url (default
    # http://localhost:30000/v1), input_key column holding the chat message list (default
    # "input"), samples per prompt (1), maximum response tokens (1024), temperature (1.0) and
    # top-p (1.0). Creates an async client with a dummy API key against the url, then, for every
    # row and every repeat index, launches one concurrent chat completion request (all rows are
    # read and all requests started at once, with no concurrency limit), waits for all, and prints
    # one json line per result in task order. Each result holds the row index, repeat index, all
    # original row fields (a row field named like the result keys would collide and raise an
    # error) and the response text of the first choice. The model name sent is a fixed dummy
    # name. Any request failure aborts the whole run.
    """
    Minimally send prompts to SGLang using OpenAI endpoints with arguments in the same format as main Miles.

    Example usage:
    python -m miles.utils.debug_utils.send_to_sglang --prompt-data /root/datasets/aime-2024/aime-2024.jsonl --input-key prompt --n-samples-per-prompt 16 --rollout-max-response-len 32768 --rollout-temperature 1 --rollout-top-p 1
    """

    async def _main_async():
        # [FUNCTION DOC] HELPER: Builds one task per (row, repeat) pair, awaits all together and
        # prints each output as a json line in task order.
        tasks = [
            asyncio.create_task(_run_one(row, row_index=row_index, repeat_index=repeat_index))
            for row_index, row in enumerate(read_file(prompt_data))
            for repeat_index in range(n_samples_per_prompt)
        ]
        outputs = await asyncio.gather(*tasks)
        for output in outputs:
            print(json.dumps(output))

    async def _run_one(row, row_index: int, repeat_index: int):
        # [FUNCTION DOC] HELPER: Sends one chat completion request with the row's messages under
        # the input key, the dummy model name and the sampling options from the enclosing scope
        # (client created after this function is defined but before it runs). Returns a mapping of
        # row index, repeat index, the row's fields and the first choice's message content.
        resp = await client.chat.completions.create(
            messages=row[input_key],
            model="dummy_model",
            max_tokens=rollout_max_response_len,
            temperature=rollout_temperature,
            top_p=rollout_top_p,
        )
        return dict(
            row_index=row_index,
            repeat_index=repeat_index,
            **row,
            response=resp.choices[0].message.content,
        )

    client = AsyncOpenAI(api_key="dummy_key", base_url=url)
    asyncio.run(_main_async())


if __name__ == "__main__":
    typer.run(main)
