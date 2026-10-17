# [MODULE DOC]
# Multi-agent math pipeline used by the multi-agent rollout example. Three agent roles all query
# the training run's inference router directly: a solver (answers the problem statement), a
# rewriter (given all solver outputs, writes a revised solution) and a selector (given the revised
# solutions, judges which is best). Pipeline (run_agent_system): run N solvers in parallel
# (N from the copied launch args), score every solver sample with the batched reward model, then
# run N rewriters over the successful solver texts, score them, run one selector over the successful
# rewrites, map its judgement back to a rewriter sample's reward, and finally scale the rewards of
# all collected samples by a correct-answer weight when the selected answer is correct (reward
# exactly one) or an incorrect-answer weight otherwise. The pipeline returns every generated training
# sample from all stages. Shared state travels on a deep-copied args object: the current base
# sample, and a results mapping with lists of samples for solver, rewriter and selector.
# Error handling is lenient: generation errors are printed and produce none.
import asyncio
import re
import time
import traceback
from copy import deepcopy

from miles.rollout.rm_hub import batched_async_rm
from miles.utils.http_utils import post
from miles.utils.types import Sample

from .prompts import SOLVER_PROMPT_TEMPLATE, generate_rewriter_template, generate_select_template


async def generate_response(args, prompt, key):
    # [FUNCTION DOC] Sends one prompt to the inference router and records the generated sample.
    # Uses fields placed on the args object by the caller: sampling parameters, tokenizer, maximum
    # context length, the template sample and the results mapping. Steps: copy the base sample, set
    # its prompt, tokenise the prompt without special tokens, and cap the new token budget at the
    # smaller of the configured maximum new tokens and the remaining context; if the remaining
    # budget is not positive returns none without recording anything. Posts token ids and the
    # sampling parameters to the router's generate endpoint requesting log probabilities. Response
    # tokens are taken from the output log probability entries (second element of each), or empty if
    # absent (an abort). The sample gets the combined tokens, the response length incremented by
    # the number of new tokens and the response text; status becomes truncated for finish reason
    # length and completed for stop (other reasons leave the status unchanged). The sample is
    # appended to the results list for the given key. The returned text is the part after the
    # reasoning end marker, stripped, only when the output (with a user marker removed) contains the
    # marker exactly once and non-empty text follows it; in that case the sample also records the
    # stripped reasoning and response content. In every other case both fields are set to none and
    # none is returned. Any exception is printed and returns none.
    # KNOWN ISSUE: because every exception is swallowed here and returned as none, the retry loops
    # in the callers never retry; the sample is also recorded before the response parsing so
    # unparseable outputs still produce training samples.
    try:
        sampling_params = args.sampling_params
        tokenizer = args.tokenizer
        max_context_length = args.rollout_max_context_len
        sample = deepcopy(args.sample)

        url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

        sample.prompt = prompt
        prompt_token_ids = tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
        sample.tokens = prompt_token_ids
        prompt_length = len(prompt_token_ids)
        current_sampling_params = deepcopy(sampling_params)
        current_sampling_params["max_new_tokens"] = min(
            sampling_params["max_new_tokens"], max_context_length - prompt_length
        )

        if current_sampling_params["max_new_tokens"] <= 0:
            return None

        payload = {"input_ids": prompt_token_ids, "sampling_params": current_sampling_params, "return_logprob": True}

        output = await post(url, payload)

        # Extract new response tokens
        if "output_token_logprobs" in output["meta_info"]:
            new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        else:
            # abort
            new_response_tokens = []

        # Update sample with tokens directly - avoiding re-tokenization
        sample.tokens = sample.tokens + new_response_tokens
        sample.response_length += len(new_response_tokens)
        sample.response = output["text"]

        match output["meta_info"]["finish_reason"]["type"]:
            case "length":
                sample.status = Sample.Status.TRUNCATED
            # case "abort":
            #     sample.status = Sample.Status.ABORTED
            case "stop":
                sample.status = Sample.Status.COMPLETED

        args.results_dict[key].append(sample)

        final = output["text"].replace("<|user|>", "")
        if "</think>" in final:
            contents = final.split("</think>")
            if len(contents) == 2 and contents[1] != "":
                reason_content = contents[0].strip()
                response_content = contents[1].strip()
                sample.reason_content = reason_content
                sample.response_content = response_content
                return response_content
        sample.reason_content = None
        sample.response_content = None
        return None
    except Exception as e:
        print(f"Error generating response: {e}")
        return None


class Agent:
    # [CLASS DOC]
    # Base class of the three roles; stateless. Provides the shared run method that queries the
    # model with retries.
    """A base class for our AI agents."""

    def __init__(self):
        # [FUNCTION DOC] No-op initialiser.
        pass

    async def run(self, args, prompt, max_retries: int = 1, key: str = None) -> str:
        # [FUNCTION DOC] Queries the model for the prompt and returns the extracted response text or
        # none. Tries up to max_retries times (default one); an exception from the generation call
        # is printed and followed by a blocking one second sleep before the next attempt; after all
        # attempts prints a failure message and returns none. The key names which results list the
        # sample is recorded in.
        # KNOWN ISSUE: the generation function never raises, so a none result is returned at once
        # and no retry ever happens; the sleep call also blocks the event loop.
        """Runs the agent by sending a prompt to the LLM."""
        for _i in range(max_retries):
            try:
                response = await generate_response(args, prompt, key=key)
                return response
            except Exception as e:
                print(f"Error querying LLM: {e}")
                time.sleep(1)
        print(f"Failed to query LLM after {max_retries} retries")
        return None


class SolverAgent(Agent):
    # [CLASS DOC]
    # Solver role: answers the problem statement directly.
    """The agent responsible for generating and improving solutions."""

    def __init__(self):
        # [FUNCTION DOC] Calls the base initialiser.
        super().__init__()

    async def generate_initial_solution(self, args, problem_statement) -> str:
        # [FUNCTION DOC] Formats the solver template with the problem statement and queries the
        # model with up to three attempts, recording the sample under the solver key. Returns the
        # response text or none.
        """Generates the first solution attempt."""
        prompt = SOLVER_PROMPT_TEMPLATE.format(problem_statement=problem_statement)
        return await self.run(args, prompt, max_retries=3, key="solver")


class RewriterAgent(Agent):
    # [CLASS DOC]
    # Rewriter role: revises a solution given all previous solver outputs.
    """The agent responsible for rewriting solutions."""

    def __init__(self):
        # [FUNCTION DOC] Calls the base initialiser.
        super().__init__()

    async def rewrite(self, args, problem_statement, previous_solutions: list[str]) -> str:
        # [FUNCTION DOC] Builds the rewriter template for the number of previous solutions, fills
        # in the problem statement and each solution under its one-based number, and queries the
        # model once, recording the sample under the rewriter key. Returns the response text or none.
        """Generates the rewrited solution."""

        # 动态生成模板
        template = generate_rewriter_template(len(previous_solutions))

        # 构建参数字典
        format_params = {"problem_statement": problem_statement}
        for i, solution in enumerate(previous_solutions):
            format_params[f"solution{i+1}"] = solution

        prompt = template.format(**format_params)
        return await self.run(args, prompt, max_retries=1, key="rewriter")


class SelectorAgent(Agent):
    # [CLASS DOC]
    # Selector role: judges which of several candidate solutions is best.
    """The agent responsible for selecting solutions."""

    def __init__(self):
        # [FUNCTION DOC] Calls the base initialiser.
        super().__init__()

    async def select(self, args, problem_statement, candidate_solutions: list[str]) -> str:
        # [FUNCTION DOC] Builds the selector template for the number of candidates, fills in the
        # problem statement and numbered candidates, and queries the model with up to ten attempts,
        # recording the sample under the selector key. Returns the judgement text or none.
        # KNOWN ISSUE: the existing docstring below was copied from the rewriter and wrongly says it
        # generates a rewritten solution.
        """Generates the rewrited solution."""

        # 动态生成模板
        template = generate_select_template(len(candidate_solutions))

        # 构建参数字典
        format_params = {"problem_statement": problem_statement}
        for i, solution in enumerate(candidate_solutions):
            format_params[f"solution{i+1}"] = solution

        prompt = template.format(**format_params)
        return await self.run(args, prompt, max_retries=10, key="selector")

    def extract_selected_solution_idx(self, response: str, candidate_solutions: list[str]) -> int:
        # [FUNCTION DOC] Parses the selector's judgement. Finds all occurrences of the word
        # "Judgment:" followed by optional whitespace and digits, takes the first, converts it from
        # one-based to a zero-based index and returns it if it lies within the candidate list; returns
        # none if out of range, if there is no match (the failure is printed) or on any parse error.
        # KNOWN ISSUE: the first match is used although the prompt asks for the judgement at the
        # end of the response, and the declared return type omits none.
        """Extracts the selected solution ID from the response."""
        PATTERN = re.compile(r"Judgment:\s*(\d+)")
        matched = PATTERN.findall(response)
        try:
            selected_id = int(matched[0]) - 1
            if selected_id < len(candidate_solutions) and selected_id >= 0:
                return selected_id
            else:
                return None
        except Exception as e:
            print(f"extract_selected_solution_idx error: {e}")
            return None


async def rewrite_worker(args, previous_solutions, problem_statement, worker_id):
    # [FUNCTION DOC] HELPER: One parallel rewriter task: creates a rewriter and returns its
    # rewrite of the problem given all previous solutions (text or none). The worker id is unused.
    rewriter = RewriterAgent()
    new_solution = await rewriter.rewrite(args, problem_statement, previous_solutions)
    return new_solution


async def solver_worker(args, problem_statement, worker_id):
    # [FUNCTION DOC] HELPER: One parallel solver task: creates a solver and returns its first
    # solution (text or none). Any exception is printed together with the worker id and traceback
    # and none is returned.
    """
    单组 solver 流程。
    """

    try:
        solver = SolverAgent()
        current_solution = await solver.generate_initial_solution(args, problem_statement)
        return current_solution

    except Exception as e:
        print(f"[Worker-{worker_id}] exception: {e}")
        print(f"[Worker-{worker_id}] traceback: {traceback.format_exc()}")
        return None


async def run_agent_system(args, sample):
    # [FUNCTION DOC] Runs the whole solve, rewrite, select pipeline for one base sample and returns
    # the list of all samples generated. Works on a deep copy of the launch args (the caller mutates
    # args) with the base sample and fresh results lists attached. Solver stage: runs the configured
    # number of parallel solvers over the sample's prompt; every recorded solver sample receives a
    # reward from the batched reward model; the successful solver texts (strings) become the previous
    # solutions. If there are none, solver sample rewards are multiplied by the incorrect-answer
    # weight and only solver samples are returned. Rewriter stage: the same number of rewriters;
    # rewriter samples are scored; the successful rewrites are collected; if none, solver and
    # rewriter rewards are scaled by the incorrect weight and both lists returned. Selector stage:
    # one selector over the rewrites; if it recorded no sample, same incorrect-weight handling as
    # above; the selector must have recorded exactly one sample (assertion otherwise). The selector
    # reward is zero if it produced no response or an unparseable or out-of-range index; otherwise
    # the chosen rewrite text is matched, by substring, against the rewriter samples' response
    # contents in order and the selector takes the reward of the first match. Finally, if the
    # selector reward is exactly one, all solver, rewriter and selector rewards are multiplied by the
    # correct-answer weight, else by the incorrect-answer weight. Returns solver, rewriter and
    # selector samples concatenated in that order.
    # KNOWN ISSUE: if no rewriter sample matches the chosen rewrite, the selector sample keeps its
    # unset default reward and the later multiplication of the selector reward by a weight can
    # fail. Also, the inner loop variables shadow the sample parameter (harmless after its use).
    """
    并发运行 num_parallel 组 pipeline。
    """

    args = deepcopy(args)  # 深拷贝 args，因为 args 在 rollout_with_multi_agents 中会被修改
    args.sample = sample
    args.results_dict = {"solver": [], "rewriter": [], "selector": []}

    problem_statement = sample.prompt
    tasks = [solver_worker(args, problem_statement, worker_id) for worker_id in range(args.num_parallel)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    rewards = await batched_async_rm(args, args.results_dict["solver"])
    for sample, reward in zip(args.results_dict["solver"], rewards, strict=False):
        sample.reward = reward

    previous_solutions = [item for item in results if isinstance(item, str)]

    def reward_adjustment(samples, reward_weight):
        # [FUNCTION DOC] HELPER: Multiplies every sample's reward by the weight in place and
        # returns the same list.
        for sample in samples:
            sample.reward = sample.reward * reward_weight
        return samples

    if len(previous_solutions) == 0:
        reward_adjustment(args.results_dict["solver"], args.incorrect_reward_weight)
        return args.results_dict["solver"]

    # Rewriting
    tasks = [
        rewrite_worker(args, previous_solutions, problem_statement, worker_id)
        for worker_id in range(args.num_parallel)
    ]
    rewrited_solutions_raw = await asyncio.gather(*tasks, return_exceptions=True)

    # 处理异常结果
    rewrited_solutions = []
    for _i, result in enumerate(rewrited_solutions_raw):
        if isinstance(result, str):
            rewrited_solutions.append(result)

    rewards = await batched_async_rm(args, args.results_dict["rewriter"])
    for sample, reward in zip(args.results_dict["rewriter"], rewards, strict=False):
        sample.reward = reward

    if len(rewrited_solutions) == 0:
        reward_adjustment(args.results_dict["solver"], args.incorrect_reward_weight)
        reward_adjustment(args.results_dict["rewriter"], args.incorrect_reward_weight)
        return args.results_dict["solver"] + args.results_dict["rewriter"]

    # Selection
    selector = SelectorAgent()
    response = await selector.select(args, problem_statement, rewrited_solutions)
    if len(args.results_dict["selector"]) == 0:
        reward_adjustment(args.results_dict["solver"], args.incorrect_reward_weight)
        reward_adjustment(args.results_dict["rewriter"], args.incorrect_reward_weight)
        return args.results_dict["solver"] + args.results_dict["rewriter"]

    assert (
        len(args.results_dict["selector"]) == 1
    ), f"selector should only return one solution, but got {len(args.results_dict['selector'])}"
    if response is None:
        args.results_dict["selector"][0].reward = 0
    else:
        selected_solution_idx = selector.extract_selected_solution_idx(response, rewrited_solutions)
        if selected_solution_idx is None:
            args.results_dict["selector"][0].reward = 0
        else:
            selected_solution = rewrited_solutions[selected_solution_idx]
            for sample in args.results_dict["rewriter"]:
                if sample.response_content is not None and selected_solution in sample.response_content:
                    args.results_dict["selector"][0].reward = sample.reward
                    break

    ## 如果最终答案正确，对所有reward添加正向的奖励；如果最终答案不正确，对所有reward添加负向的惩罚。
    if args.results_dict["selector"][0].reward == 1:
        reward_adjustment(args.results_dict["solver"], args.correct_reward_weight)
        reward_adjustment(args.results_dict["rewriter"], args.correct_reward_weight)
        reward_adjustment(args.results_dict["selector"], args.correct_reward_weight)
    else:
        reward_adjustment(args.results_dict["solver"], args.incorrect_reward_weight)
        reward_adjustment(args.results_dict["rewriter"], args.incorrect_reward_weight)
        reward_adjustment(args.results_dict["selector"], args.incorrect_reward_weight)

    return args.results_dict["solver"] + args.results_dict["rewriter"] + args.results_dict["selector"]
