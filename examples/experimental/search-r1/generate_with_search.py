# [MODULE DOC] Custom generate and reward functions for the Search-R1 example: multi-turn
# question answering in which the model emits search queries between search tags, the module runs
# the query against a backend (a local retrieval server or the Google search service, chosen by a
# module-level configuration mapping), and appends the retrieved passages between information tags
# as masked-out observation tokens before the model continues. Configuration mapping keys:
# maximum turns (2), top-k passages (3), search concurrency (256, a module-level semaphore),
# backend (local by default), per-backend settings (url or api key, snippet-only flag, proxy), a
# flag to collect rollout log probabilities (true) and a format score (0.2). The generate function
# talks directly to the SGLang router over HTTP using text prompts; the reward function scores
# with the exact-match scorer from the sibling format module.
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/ceee7b89655ed52f205b9beb98e1190c3eedcfb0/search_r1/llm_agent/generation.py
# This is a unified version supporting both local search and Google search, with optional log probability collection

import asyncio
import re

from qa_em_format import compute_score_em

from miles.rollout.sglang_rollout import GenerateState
from miles.utils.http_utils import post
from miles.utils.types import Sample

# Configuration for Search-R1
SEARCH_R1_CONFIGS = {
    # ============== General Configuration ==============
    "max_turns": 2,
    "topk": 3,
    "search_concurrency": 256,
    # ============== Search Backend Selection ==============
    "search_backend": "local",  # Options: "local" or "google"
    # ============== Local Search Configuration ==============
    # (Only used when search_backend="local")
    "local": {
        "search_url": "http://127.0.0.1:8000/retrieve",  # URL of your local retrieval server
        "proxy": None,  # Set to your proxy if needed
    },
    # ============== Google Search Configuration ==============
    # (Only used when search_backend="google")
    "google": {
        "api_key": "your_api_key_here",  # Replace with your actual API key
        "snippet_only": True,  # Set to True to only return snippets
        "proxy": None,  # Set to your proxy if needed
    },
    # ============== Log Probability Collection ==============
    "return_logprob": True,  # Set to True to collect log probabilities for TIS metrics
    # ============== Reward Model Configuration ==============
    "format_score": 0.2,
}


SEMAPHORE = asyncio.Semaphore(SEARCH_R1_CONFIGS["search_concurrency"])


def _passages2string(retrieval_result):
    # [FUNCTION DOC] HELPER: Formats retrieved documents into one reference text. Each item is a
    # mapping with a "document" mapping holding "contents"; the first line of the contents is the
    # title and the remaining lines are the body. Output has one line per document in the form
    # "Doc <1-based index>(Title: <title>) <body>" followed by a newline. Returns an empty string for
    # no results; a missing key raises a key error.
    """
    Convert retrieval results to a formatted string.
    This function works with both google_search and local_search results.
    """
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

    return format_reference


async def search(query: str) -> str:
    # [FUNCTION DOC] Runs one search and returns the formatted reference text. Reads the backend
    # name from the configuration mapping at call time. Local: lazily imports the local search
    # helper and calls it with the configured url, the query, top-k and proxy. Google: lazily
    # imports the Google helper and calls it with the api key, query, top-k, snippet-only flag and
    # proxy. Any other backend raises a value error. The retrieved documents are then converted by
    # the passage formatter. Network side effects.
    """
    Perform search using either local search engine or Google search.
    The search backend is determined by SEARCH_R1_CONFIGS["search_backend"].
    """
    backend = SEARCH_R1_CONFIGS["search_backend"]

    if backend == "local":
        from local_search_server import local_search

        local_config = SEARCH_R1_CONFIGS["local"]
        result = await local_search(
            local_config["search_url"],
            query,
            SEARCH_R1_CONFIGS["topk"],
            proxy=local_config["proxy"],
        )
    elif backend == "google":
        from google_search_server import google_search

        google_config = SEARCH_R1_CONFIGS["google"]
        result = await google_search(
            google_config["api_key"],
            query,
            SEARCH_R1_CONFIGS["topk"],
            snippet_only=google_config["snippet_only"],
            proxy=google_config["proxy"],
        )
    else:
        raise ValueError(f"Unknown search backend: {backend}. " f"Must be either 'local' or 'google'.")

    return _passages2string(result)


# IMPORTANT: When we need to collect log probabilities (logp), we CANNOT do any postprocessing
# on the strings returned from the inference engine (sglang). This is because:
# 1. We don't know how to truncate the corresponding tokens/logp arrays to match the modified string
# 2. Re-tokenizing the postprocessed string may produce different tokens than what the engine generated,
#    leading to misalignment between tokens and their log probabilities
# Therefore, postprocess_responses is only used when return_logprob=False.
def postprocess_responses(resp: str) -> str:
    # [FUNCTION DOC] HELPER: Truncates a model response right after the first closing search tag if
    # present; otherwise right after the first closing answer tag if present; otherwise returns it
    # unchanged. Only valid when log probabilities are not collected (the text would no longer
    # match the generated tokens).
    """
    Post-process response to ensure tag completeness.
    Only used when SEARCH_R1_CONFIGS["return_logprob"] is False.
    """
    return (
        resp.split("</search>")[0] + "</search>"
        if "</search>" in resp
        else resp.split("</answer>")[0] + "</answer>" if "</answer>" in resp else resp
    )


def postprocess_predictions(prediction: str):
    # [FUNCTION DOC] HELPER: Finds the first search-or-answer tagged span in the text (multi-line,
    # shortest match, same tag name closing it). Returns a pair of action name ("search" or
    # "answer") and the trimmed inner text; (none, empty string) if no span is found.
    pattern = r"<(search|answer)>(.*?)</\1>"
    match = re.search(pattern, prediction, re.DOTALL)
    if match:
        content = match.group(2).strip()  # Return only the content inside the tags
        action = match.group(1)
    else:
        content = ""
        action = None

    return action, content


async def execute_predictions(prediction: str) -> str:
    # [FUNCTION DOC] HELPER: Interprets one model turn and returns a pair of next observation text
    # and a done flag. Search action: runs the search for the query while holding the module
    # concurrency semaphore and returns the trimmed results wrapped in information tags with blank
    # lines around, not done. Answer action: empty observation, done. No valid action: a fixed
    # reminder message explaining the tag formats, not done.
    # KNOWN ISSUE: the return annotation says a string but the function returns a pair.
    action, content = postprocess_predictions(prediction)

    if action == "search":
        search_query = content
        async with SEMAPHORE:
            search_results = await search(search_query)
        next_obs = f"\n\n<information>{search_results.strip()}</information>\n\n"
        done = False
    elif action == "answer":
        next_obs = ""
        done = True
    else:
        next_obs = "\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n"
        done = False

    return next_obs, done


async def generate(args, sample: Sample, sampling_params) -> Sample:
    # [FUNCTION DOC] Custom generate function: runs the multi-turn search loop for one sample and
    # returns the same sample object filled in. Asserts partial rollout is off. Builds the router
    # generate url from args, tokenizes the prompt text (no special tokens) and starts with empty
    # response, token ids, loss mask and (if collecting) log prob lists. For up to max-turns turns:
    # post the prompt plus the response so far as text with the given sampling params (and the
    # log-prob request flag when collecting); if the finish reason is abort, mark the sample
    # aborted and return immediately. When collecting log probs, raises a runtime error if the
    # engine omitted per-token log probs, and takes token ids and log probs directly from them (so
    # they stay aligned); otherwise truncates the text with the response post-processor and
    # re-tokenizes it. Appends the turn text, tokens, loss mask ones, and log probs. Stops if the
    # finish reason is length. Otherwise executes the turn: done ends the loop; else (asserting a
    # non-empty observation) tokenizes the observation, appends it to the response and tokens with
    # loss mask zeros and zero placeholder log probs, and asserts token and log-prob lengths match.
    # After the loop sets tokens (prompt + response), response length, response text, loss mask and
    # prompt, sets rollout log probs when collecting (none if the list is empty), and sets the status
    # from the last finish reason: length -> truncated, abort -> aborted, stop -> completed (any
    # other value leaves the status unchanged).
    # KNOWN ISSUE: a comment mentions handling partial rollout continuation although partial
    # rollout is rejected; on the last allowed turn a search is still executed and its observation
    # appended although no further generation follows; a sample that exhausts the turn limit gets
    # whatever status the last engine finish reason maps to (normally completed).
    assert not args.partial_rollout, "Partial rollout is not supported for this function at the moment."

    state = GenerateState(args)

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Handle partial rollout samples: continue generation from existing response
    prompt_text = sample.prompt
    prompt_tokens_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response = ""
    response_token_ids = []
    loss_mask = []
    rollout_log_probs = [] if SEARCH_R1_CONFIGS["return_logprob"] else None

    for _turn_idx in range(SEARCH_R1_CONFIGS["max_turns"]):
        payload = {
            "text": prompt_text + response,
            "sampling_params": sampling_params,
        }
        # Add log probability collection if enabled
        if SEARCH_R1_CONFIGS["return_logprob"]:
            payload["return_logprob"] = True

        output = await post(url, payload)

        # abort
        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        cur_response = output["text"]

        # Extract tokens and log probs based on configuration
        if SEARCH_R1_CONFIGS["return_logprob"]:
            # Extract log probs from output - required for TIS metrics
            if "output_token_logprobs" not in output["meta_info"]:
                raise RuntimeError(
                    "output_token_logprobs not found in output meta_info. "
                    "Make sure 'return_logprob': True is set in the payload."
                )

            # Use token IDs and log probs directly from output_token_logprobs
            # This ensures perfect alignment between tokens and log probs
            # output_token_logprobs format: [[log_prob, token_id, ...], ...]
            cur_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            cur_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        else:
            # When not collecting log probs, we can safely postprocess the response
            cur_response = postprocess_responses(cur_response)
            # Tokenize the (possibly postprocessed) response
            cur_response_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]

        response += cur_response
        response_token_ids += cur_response_token_ids
        loss_mask += [1] * len(cur_response_token_ids)

        # Add log probs if enabled
        if SEARCH_R1_CONFIGS["return_logprob"]:
            rollout_log_probs += cur_response_log_probs

        if output["meta_info"]["finish_reason"]["type"] == "length":
            break

        next_obs, done = await execute_predictions(cur_response)
        if done:
            break

        assert next_obs != "", "Next observation should not be empty."
        obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
        response += next_obs
        response_token_ids += obs_tokens_ids
        loss_mask += [0] * len(obs_tokens_ids)

        # Add dummy log probs for observation tokens if enabled (they won't be used due to loss_mask=0)
        if SEARCH_R1_CONFIGS["return_logprob"]:
            rollout_log_probs += [0.0] * len(obs_tokens_ids)

            # Verify alignment when collecting log probs
            assert len(response_token_ids) == len(
                rollout_log_probs
            ), f"Token/logp length mismatch: {len(response_token_ids)} tokens vs {len(rollout_log_probs)} logps"

    # Store statistics for wandb logging
    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_mask
    sample.prompt = prompt_text

    # Store log probs if enabled
    if SEARCH_R1_CONFIGS["return_logprob"]:
        sample.rollout_log_probs = rollout_log_probs if rollout_log_probs else None

    match output["meta_info"]["finish_reason"]["type"]:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED

    return sample


async def reward_func(args, sample, **kwargs):
    # [FUNCTION DOC] Custom reward hook for one sample. Raises a type error if the argument is not a
    # sample object. Returns the exact-match score of the full text (prompt plus response) against
    # the ground truth stored in the sample label under "ground_truth", passing the configured
    # format score.
    # KNOWN ISSUE: the scorer ignores its format-score argument (only its structure/final/retrieval
    # scores, all zero by default, are used), so the configured value has no effect and the reward
    # is effectively 1.0 for a correct well-formed answer and 0 otherwise.
    """The reward function for retrieval-based question answering.

    Args:
        args: the arguments
        sample: the sample to evaluate
    """
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    score = compute_score_em(
        solution_str=sample.prompt + sample.response,
        ground_truth=sample.label["ground_truth"],
        format_score=SEARCH_R1_CONFIGS["format_score"],
    )

    return score
