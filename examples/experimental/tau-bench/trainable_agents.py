# [MODULE DOC] Trainable tau-bench agent for Miles rollouts. Defines the interaction status enum
# and result dataclass; a function turning a parsed tool call into a tau-bench action; the tool
# instruction text that replaces the default multi-call sentence in the chat template output; a
# mixin that runs a multi-turn conversation between the model (via the SGLang router's text
# generate endpoint) and a tau-bench environment while tracking token ids and a loss mask (model
# tokens trained on, environment tokens masked out); a tool-calling agent class combining the mixin
# with the tau-bench tool-calling agent; and a factory choosing the agent from the run config.
import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from openai_tool_adapter import create_openai_adapter
from tau_bench.agents.base import Agent
from tau_bench.agents.tool_calling_agent import RESPOND_ACTION_NAME, ToolCallingAgent
from tau_bench.types import Action, RunConfig
from transformers import AutoTokenizer

from miles.rollout.sglang_rollout import GenerateState
from miles.utils.http_utils import post

# Set up logger for this module
logger = logging.getLogger(__name__)


class Status(Enum):
    # [CLASS DOC] Outcome of an interaction: completed (environment finished), truncated (step
    # limit reached without finishing) or aborted (generation abort, unparsable output or
    # environment failure). Values are the lowercase names.
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ABORTED = "aborted"


@dataclass
class InteractionResult:
    # [CLASS DOC] Result container for one interaction: initial prompt text, final reward, the
    # message list, an info mapping, the concatenated assistant response text, the loss mask list
    # (one entry per response token), the full token id list (prompt then response) and the status
    # (default completed). A response_length attribute is added dynamically when the result is
    # finalized.
    # KNOWN ISSUE: the tokens field is annotated as an integer but holds a list of integers, and
    # response_length is not a declared field.
    prompt: str
    reward: float
    messages: list[dict[str, Any]]
    info: dict[str, Any]
    response: str = ""
    loss_mask: list[int] | None = None
    tokens: int | None = None
    status: Status = Status.COMPLETED


def call_to_action_sglang(calls: list[Any], text_response: str) -> Action:
    # [FUNCTION DOC] Converts parsed tool calls into a tau-bench action. Default: the respond action
    # with the text as content. If there are calls, only the first is used (extra calls logged at
    # debug level): its parameters text is decoded as JSON and, if the result is a mapping, the
    # action becomes that tool name with the mapping as keyword arguments; a non-mapping logs a
    # warning and keeps the default. A missing key or invalid JSON raises (not caught here).
    # KNOWN ISSUE: duplicates the adapter module's method of the same purpose, which additionally
    # catches invalid JSON; here invalid JSON propagates and ends the interaction with an exception.
    """
    Convert sglang response message to Action, similar to original message_to_action
    but adapted for sglang response format.
    """
    # Default action if no action was found.
    action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": text_response})
    if calls:
        if len(calls) > 1:
            logger.debug("Multiple tool calls identified, only taking first.")
        tool_call = calls[0]
        params = json.loads(tool_call["parameters"])
        if not isinstance(params, dict):
            logger.warning(f"{params} does not follow dict structure for action")
        else:
            action = Action(name=tool_call["name"], kwargs=params)
    return action


TOOL_INSTRUCTION = (
    " At each turn, you are allowed to call one or no function to assist "
    "with task execution using <tools></tools> XML tags.\n"
    "YOU MUST EXECUTE TOOLS TO MAKE ANY MODIFICATIONS OR CANCELLATIONS. "
    "Each tool call leads to a message returned by the system.\n"
    "NEVER confirm execution to the user without seeing confirmation "
    "from the tool system.\n"
)


class TrainableAgentMixin:
    # [CLASS DOC] Mixin adding the asynchronous training interaction to a tau-bench agent. The host
    # class must provide the wiki text, the tool descriptions, an OpenAI adapter, and the agent
    # temperature. Methods are small steps used by the main solve coroutine.
    """
    Mixin class that provides trainable agent functionality for tau-bench environments.

    This mixin extends the original tau-bench agent with async LLM interaction
    capabilities for reinforcement learning training using sglang servers.
    """

    def _reformulate_tool_call(self, text: str) -> str:
        # [FUNCTION DOC] HELPER: Replaces the sentence "You may call one or more functions to assist
        # with the user query." (the default tool template text) with the module's tool instruction
        # (at most one call per turn, must execute tools to modify, never confirm without tool
        # confirmation). Returns the new text; unchanged if the sentence is absent.
        """
        Reformulate tool call instruction for tau-bench environment.

        The default tool template assumes one or more function calls, but for
        tau-bench, at most one tool call or skip tool calls are the valid options.

        Args:
            text: Original tool instruction text

        Returns:
            Reformulated tool instruction text
        """
        return text.replace("You may call one or more functions to assist with the user query.", TOOL_INSTRUCTION)

    async def _call_llm(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        # [FUNCTION DOC] HELPER: Posts the payload to the url with the shared Miles HTTP helper and
        # returns the decoded response mapping.
        """
        Make an LLM call tracking.

        Args:
            url: SGLang server URL
            payload: Request payload containing text and sampling parameters

        Returns:
            LLM response from sglang server
        """
        return await post(url, payload)

    def _parse_tool(self, response: str) -> dict[str, Any]:
        # [FUNCTION DOC] HELPER: Delegates to the OpenAI adapter's response parser and returns its
        # result mapping (success flag, parsed result, message or error).
        """
        Parse tool calls from LLM response string.

        Args:
            response: Raw response text from sglang

        Returns:
            Parsed tool call result in OpenAI format
        """
        return self.openai_adapter.parse_response_to_openai_format(response)

    async def _execute_tool(self, env, action: Action):
        # [FUNCTION DOC] HELPER: Steps the environment with the action (a blocking synchronous call
        # inside a coroutine) and returns the step result (observation, reward, done, info).
        """
        Execute a tool/action in the environment.

        Args:
            env: Tau-bench environment instance
            action: Action to execute

        Returns:
            Environment step result
        """
        return env.step(action)

    def _initialize_environment(self, env, task_index: int | None) -> tuple[str, dict[str, Any]]:
        # [FUNCTION DOC] HELPER: Resets the environment (to the given task index if not none) and
        # returns a pair of the first observation text and the reset info dumped to a mapping.
        """
        Initialize the environment and get initial observation.

        Args:
            env: Tau-bench environment instance
            task_index: Task index to reset to

        Returns:
            Tuple of (observation, info)
        """
        if task_index is not None:
            env_reset_res = env.reset(task_index=task_index)
        else:
            env_reset_res = env.reset()
        return env_reset_res.observation, env_reset_res.info.model_dump()

    def _build_initial_messages(self, obs: str) -> list[dict[str, Any]]:
        # [FUNCTION DOC] HELPER: Returns a two-message list: a system message with the agent's wiki
        # text and a user message with the first observation.
        """
        Build initial conversation messages.

        Args:
            obs: Initial observation from environment

        Returns:
            List of initial messages
        """
        return [{"role": "system", "content": self.wiki}, {"role": "user", "content": obs}]

    def _prepare_prompt_tokens(self, state: GenerateState, messages: list[dict[str, Any]]) -> tuple[str, list[int]]:
        # [FUNCTION DOC] HELPER: Renders the messages with the tokenizer's chat template as text
        # (adding the generation prompt and passing the tool descriptions), applies the tool
        # instruction rewrite, and tokenizes the text without special tokens. Returns the pair of
        # prompt text and token ids.
        """
        Prepare prompt text and tokenize it.

        Args:
            state: GenerateState instance with tokenizer
            messages: Conversation messages

        Returns:
            Tuple of (prompt_text, prompt_token_ids)
        """
        prompt_text = state.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, tools=self.tools_info
        )
        # Reformulate tool call instruction for tau-bench
        prompt_text = self._reformulate_tool_call(prompt_text)
        prompt_token_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        return prompt_text, prompt_token_ids

    async def asolve(
        self,
        env,
        rollout_args: dict[str, Any],
        sampling_params: dict[str, Any],
        task_index: int | None = None,
        max_num_steps: int = 30,
    ) -> InteractionResult:
        # [FUNCTION DOC] Runs one full episode and returns the interaction result. Uses attribute
        # access on rollout_args (router ip and port; a namespace object, not a mapping), builds the
        # router generate url, resets the environment, and starts with system (wiki) and user
        # (observation) messages and the initial prompt tokens. Loop for up to max_num_steps: render
        # the whole conversation as text with the chat template (with tools, generation prompt, tool
        # instruction rewrite) and post it with the sampling params. A finish reason of abort ends
        # the episode as aborted. The response text loses a trailing end-of-turn marker if present.
        # The response is parsed with the OpenAI adapter; a parse failure or exception ends the
        # episode as aborted. Appends the assistant message (raw text) and its token delta with loss
        # mask ones. Converts the parse to an action (first tool call, else a respond action with
        # the plain text); steps the environment (an exception ends as aborted). Appends a tool
        # message (named after the tool) for tool actions or a user message for respond actions with
        # the observation, and its token delta with loss mask zeros. Reward becomes the latest
        # environment reward (not a sum), info is merged with the step info, and a done flag ends
        # the loop as completed. If the last step was not done, status is truncated. The final
        # result is built by the helper (also used for aborts, with the data so far).
        # KNOWN ISSUE: the declared type of rollout_args is a mapping though attribute access is
        # used; if the loop body never runs (zero steps) the final check reads an undefined step
        # result and raises a name error.
        """
        Execute async agent-environment interaction for training.

        This method extends the original Agent to support async interaction with LLM
        server for reinforcement learning training. It maintains conversation history,
        tracks tokens, and records metadata for training purposes.

        Args:
            env: Tau-bench environment instance
            rollout_args: Rollout configuration arguments
            sampling_params: LLM sampling parameters
            task_index: Specific task index to solve (optional)
            max_num_steps: Maximum number of interaction steps

        Returns:
            InteractionResult containing the complete interaction trajectory
        """
        # Initialize environment and state
        state = GenerateState(rollout_args)
        url = f"http://{rollout_args.sglang_router_ip}:" f"{rollout_args.sglang_router_port}/generate"

        # Get initial environment state
        obs, info = self._initialize_environment(env, task_index)

        # Build initial conversation
        messages = self._build_initial_messages(obs)
        prompt_text, prompt_token_ids = self._prepare_prompt_tokens(state, messages)

        # Initialize tracking variables
        loss_masks = []
        response_token_ids = []
        total_reward = 0.0

        # Initialize result
        res = InteractionResult(prompt=prompt_text, reward=0, messages=[], info={})

        # Multi-turn interaction loop
        for _ in range(max_num_steps):
            # Prepare payload for sglang
            text_input = state.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, tools=self.tools_info
            )
            # Reformulate tool call instruction for tau-bench
            text_input = self._reformulate_tool_call(text_input)
            payload = {"text": text_input, "sampling_params": sampling_params}

            # Send request to sglang server
            output = await self._call_llm(url, payload)

            # Check for abort
            if output["meta_info"]["finish_reason"]["type"] == "abort":
                res.status = Status.ABORTED
                return self._build_final_result(
                    res, total_reward, info, messages, loss_masks, prompt_token_ids, response_token_ids
                )

            response = output["text"]
            # Remove end of conversation token if present
            if response.endswith("<|im_end|>"):
                response = response[:-10]

            # Parse tool calls using OpenAI adapter
            logger.debug(f"Using OpenAI adapter to parse response: {response[:100]}...")
            try:
                openai_result = self._parse_tool(response)
                logger.debug(f"OpenAI adapter result: success={openai_result['success']}")

                if not openai_result["success"]:
                    logger.warning(f"OpenAI adapter failed: {openai_result['error']}")
                    logger.warning(
                        f"rollout response: {response} can not be parsed into " f"tool calls {openai_result['error']}"
                    )
                    res.status = Status.ABORTED
                    return self._build_final_result(
                        res, total_reward, info, messages, loss_masks, prompt_token_ids, response_token_ids
                    )

                # Extract parsed results
                parsed = openai_result["parsed_result"]
                logger.debug(
                    f"Successfully parsed - normal_text: '{parsed['normal_text']}', " f"calls: {parsed['calls']}"
                )

            except Exception as e:
                logger.warning(f"Exception in OpenAI adapter: {e}")
                logger.warning(f"rollout response: {response} can not be parsed into " f"tool calls {e}")
                res.status = Status.ABORTED
                return self._build_final_result(
                    res, total_reward, info, messages, loss_masks, prompt_token_ids, response_token_ids
                )

            # Add assistant response to conversation
            messages.append({"role": "assistant", "content": response})
            assistant_token_ids, assistant_loss_mask = self._get_token_delta(state.tokenizer, messages)
            response_token_ids.extend(assistant_token_ids)
            loss_masks.extend(assistant_loss_mask)

            # Execute action in environment
            agent_content, calls = parsed["normal_text"], parsed["calls"]
            logger.debug(f"Creating action from - content: '{agent_content}', " f"calls: {calls}")
            action = call_to_action_sglang(calls, agent_content)
            logger.debug(f"Created action: {action}")

            try:
                env_response = await self._execute_tool(env, action)
            except Exception as e:
                logger.warning("Environment step failed, this is usually related to " "the User simulation call.")
                logger.warning(f"Error: {e}")
                res.status = Status.ABORTED
                return self._build_final_result(
                    res, total_reward, info, messages, loss_masks, prompt_token_ids, response_token_ids
                )

            logger.debug(f"Environment response: reward={env_response.reward}, " f"done={env_response.done}")

            # Update message history based on action type
            if action.name != RESPOND_ACTION_NAME:
                messages.append(
                    {
                        "role": "tool",
                        "name": action.name,
                        "content": env_response.observation,
                    }
                )
            else:
                # Direct response from user
                messages.append({"role": "user", "content": env_response.observation})

            # Update token tracking
            env_token_ids, env_loss_mask = self._get_token_delta(state.tokenizer, messages)
            response_token_ids.extend(env_token_ids)
            loss_masks.extend(env_loss_mask)

            # Update reward and info
            total_reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}

            # Check if done
            if env_response.done:
                res.status = Status.COMPLETED
                break

        # Handle truncation
        if not env_response.done:
            res.status = Status.TRUNCATED

        return self._build_final_result(
            res, total_reward, info, messages, loss_masks, prompt_token_ids, response_token_ids
        )

    def _get_token_delta(self, tokenizer: AutoTokenizer, messages: list[dict]) -> tuple[list[int], list[int]]:
        # [FUNCTION DOC] HELPER: Computes the new tokens contributed by the LAST message. Renders the
        # full conversation as text without a generation prompt. If the last message is from the
        # assistant, the reference is the conversation without it rendered WITH a generation prompt
        # and the new text is the remainder, tokenized without special tokens, with loss mask ones.
        # Otherwise the reference is the previous messages rendered without a generation prompt and
        # the remainder is tokenized with loss mask zeros. Returns the pair of token ids and mask.
        # Assumes the reference text is an exact prefix of the full rendering (slicing by length).
        # KNOWN ISSUE: the inline comments describe the mask semantics inversely ("mask only the new
        # assistant tokens" / "don't mask environment tokens"); the code trains on assistant tokens
        # (ones) and excludes environment tokens (zeros).
        """
        Calculate token delta for multi-turn conversations.

        Tokenization logic adapted from:
        https://verl.readthedocs.io/en/v0.4.1/sglang_multiturn/multiturn.html
        to calculate the right token count in a multi-turn environment using
        delta between messages.

        Args:
            tokenizer: Tokenizer instance
            messages: Conversation messages

        Returns:
            Tuple of (token_ids, loss_mask)
        """
        curr = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        token_ids = []
        loss_mask = []

        # Case 1: last message is an assistant response
        if messages[-1]["role"] == "assistant":
            prev = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=False)
            new_tokens = tokenizer.encode(curr[len(prev) :], add_special_tokens=False)
            token_ids += new_tokens
            loss_mask += [1] * len(new_tokens)  # Mask only the new assistant tokens
        else:
            # Case 2: last message is a tool response or environment observation
            prev = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=False, tokenize=False)
            new_tokens = tokenizer.encode(curr[len(prev) :], add_special_tokens=False)
            token_ids += new_tokens
            loss_mask += [0] * len(new_tokens)  # Don't mask environment/tool tokens

        return token_ids, loss_mask

    def _build_final_result(
        self,
        res: InteractionResult,
        total_reward: float,
        info: dict[str, Any],
        messages: list[dict[str, Any]],
        loss_masks: list[int],
        prompt_token_ids: list[int],
        response_token_ids: list[int],
    ) -> InteractionResult:
        # [FUNCTION DOC] HELPER: Fills the result object in place and returns it: reward, info,
        # messages, loss mask, tokens (prompt tokens followed by response tokens), response text (the
        # content of all assistant messages concatenated with no separator; a missing content counts
        # as empty), and response length equal to the loss mask length. Writes a debug log.
        """
        Build the final interaction result with all collected data.

        Args:
            res: InteractionResult instance to populate
            total_reward: Total reward accumulated during interaction
            info: Environment info dictionary
            messages: Complete conversation messages
            loss_masks: Loss masks for training
            prompt_token_ids: Prompt token IDs
            response_token_ids: Response token IDs

        Returns:
            Populated InteractionResult
        """
        res.reward = total_reward
        res.info = info
        res.messages = messages
        res.loss_mask = loss_masks
        res.tokens = prompt_token_ids + response_token_ids
        res.response = "".join([msg.get("content", "") for msg in messages if msg["role"] == "assistant"])
        res.response_length = len(loss_masks)

        logger.debug(
            f"_build_final_result: response_length={res.response_length}, "
            f"response_loss_mask_len={len(loss_masks)}, "
            f"prompt_token_len={len(prompt_token_ids)}, "
            f"response_token_len={len(response_token_ids)}, "
            f"response='{res.response[:100]}...'"
        )
        return res


class TrainableToolCallingAgent(ToolCallingAgent, TrainableAgentMixin):
    # [CLASS DOC] Tool-calling agent that is also trainable: inherits tau-bench behavior and the
    # asynchronous interaction mixin. Keeps default rollout args and sampling params and an OpenAI
    # adapter built from the tool descriptions with the qwen25 parser.
    """
    A trainable version of ToolCallingAgent that uses sglang rollout for training.

    This agent combines the original ToolCallingAgent functionality with the
    TrainableAgentMixin to support async interaction with sglang servers for
    reinforcement learning training.
    """

    def __init__(
        self,
        tools_info: list[dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
        rollout_args: dict[str, Any] | None = None,
        sampling_params: dict[str, Any] | None = None,
    ):
        # [FUNCTION DOC] Initializes the parent agent with tool info, wiki, model, provider and
        # temperature (default 0.0). Stores rollout args, or if none given a default mapping (router
        # at 127.0.0.1 port 30000, http2 off), and sampling params, or if none given a default
        # (the agent temperature, 512 new tokens, top-p 0.9, top-k 50). Creates the OpenAI adapter
        # with the tool descriptions and the qwen25 parser.
        # KNOWN ISSUE: the default rollout args is a plain mapping, but the solve method reads the
        # router address through attribute access, so the default would fail there.
        # Initialize the parent ToolCallingAgent
        super().__init__(
            tools_info=tools_info,
            wiki=wiki,
            model=model,
            provider=provider,
            temperature=temperature,
        )

        # Store rollout and sampling parameters as instance variables
        self.rollout_args = rollout_args or {
            "sglang_router_ip": "127.0.0.1",
            "sglang_router_port": 30000,
            "use_http2": False,
        }
        self.sampling_params = sampling_params or {
            "temperature": self.temperature,
            "max_new_tokens": 512,
            "top_p": 0.9,
            "top_k": 50,
        }
        # Initialize OpenAI adapter
        self.openai_adapter = create_openai_adapter(tools_info=self.tools_info, parser_type="qwen25")


def agent_factory(
    tools_info: list[dict[str, Any]],
    wiki,
    config: RunConfig,
    rollout_args: dict[str, Any] | None = None,
    sampling_params: dict[str, Any] | None = None,
) -> Agent:
    # [FUNCTION DOC] Factory: for the tool-calling strategy (read from the config's agent strategy
    # attribute) returns a trainable tool-calling agent built from the tool info, wiki, the config's
    # model, provider and temperature, and the given rollout args and sampling params. Any other
    # strategy raises a not-implemented error.
    # KNOWN ISSUE: the generate module's config sets a key named agent rather than agent strategy,
    # so that key is not what this function reads.
    if config.agent_strategy == "tool-calling":
        return TrainableToolCallingAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature,
            rollout_args=rollout_args,
            sampling_params=sampling_params,
        )
    else:
        raise NotImplementedError(f"Unsupported agent strategy: {config.agent_strategy}")
