# [MODULE DOC] Tool execution backend and reward function for the multi-turn tool-use example.
# Provides a single tool, a Python code interpreter, exposed to the rollout through three
# module-level names: the list of tool specifications (in the chat function-calling schema),
# an asynchronous function that runs a named tool with an argument mapping and returns a text
# result, and an asynchronous reward function. Code runs in a child Python process inside a
# temporary directory after a regular-expression safety screen, under a wall-clock timeout, with
# output captured. A module-level semaphore bounds concurrent tool executions, and the host
# process memory is checked and garbage collected around executions. A configuration mapping
# holds limits: turn and tool-call limits and a CPU limit (not used in this module), concurrency
# 32, code timeout 120 seconds, a memory-limit string for the interpreter (not enforced; see the
# sandbox), and memory thresholds in megabytes (cleanup levels 3072, 6144, 9216 and an execution
# refusal limit of 12288).
import asyncio
import gc
import os
import re
import subprocess
import tempfile
from contextlib import contextmanager
from typing import Any
import psutil

from miles.rollout.rm_hub.math_dapo_utils import compute_score as math_dapo_compute_score
from miles.utils.types import Sample

# Configuration for tool execution
TOOL_CONFIGS = {
    "max_turns": 16,
    "max_tool_calls": 16,
    "tool_concurrency": 32,  # Aggressive: 32 concurrent processes
    # Python interpreter settings
    "python_timeout": 120,  # 2 minutes for complex calculations
    "python_memory_limit": "4GB",  # 4GB per Python process
    "python_cpu_limit": 1,
    # Memory management settings
    "max_memory_usage": 12288,  # 12GB total (75% of 16GB)
    "cleanup_threshold": 6144,  # 6GB
    "aggressive_cleanup_threshold": 3072,  # 3GB
    "force_cleanup_threshold": 9216,  # 9GB
}

# Global semaphore for controlling concurrent tool executions
SEMAPHORE = asyncio.Semaphore(TOOL_CONFIGS["tool_concurrency"])


def get_memory_usage() -> float:
    # [FUNCTION DOC] HELPER: Resident memory of the current process in megabytes (float).
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def cleanup_memory():
    # [FUNCTION DOC] HELPER: Run one full garbage collection pass.
    """Force garbage collection to free memory"""
    gc.collect()


def aggressive_cleanup_memory():
    # [FUNCTION DOC] HELPER: Run three garbage collection passes, then for each of a few common
    # scientific libraries that is already loaded, call a cache-clearing method if the module
    # has one. In practice those modules have no such method, so this only collects garbage.
    """More aggressive memory cleanup"""
    # Force multiple garbage collection cycles
    for _ in range(3):
        gc.collect()

    # Clear Python's internal caches
    import sys

    # Note: sys.intern doesn't have a clear method, so we skip this
    # Clear module cache if possible
    if hasattr(sys, "modules"):
        # Don't clear all modules, but clear some common ones that might cache data
        modules_to_clear = ["numpy", "pandas", "matplotlib", "scipy"]
        for module_name in modules_to_clear:
            if module_name in sys.modules:
                module = sys.modules[module_name]
                if hasattr(module, "clear_cache"):
                    module.clear_cache()


def check_and_cleanup_memory():
    # [FUNCTION DOC] HELPER: Measure process memory and clean up according to tiers; returns a
    # message string describing what happened, or none if below the lowest threshold. Above 9216
    # MB: aggressive cleanup and a warning message. Above 6144 MB: one garbage collection and an
    # info message. Above 3072 MB: one garbage collection and a "light cleanup" message.
    # KNOWN ISSUE: the configuration key named aggressive threshold holds the LOWEST level and
    # leads only to light cleanup, while the key named force threshold is the highest, so the
    # key names do not match the behavior; the normal and light branches both just run one
    # garbage collection.
    """Check memory usage and perform appropriate cleanup"""
    current_memory = get_memory_usage()

    if current_memory > TOOL_CONFIGS["force_cleanup_threshold"]:
        # Force aggressive cleanup
        aggressive_cleanup_memory()
        return f"Warning: High memory usage ({current_memory:.1f}MB), performed aggressive cleanup"
    elif current_memory > TOOL_CONFIGS["cleanup_threshold"]:
        # Normal cleanup
        cleanup_memory()
        return f"Info: Memory usage ({current_memory:.1f}MB), performed cleanup"
    elif current_memory > TOOL_CONFIGS["aggressive_cleanup_threshold"]:
        # Light cleanup
        gc.collect()
        return f"Info: Memory usage ({current_memory:.1f}MB), performed light cleanup"

    return None


class PythonSandbox:
    # [CLASS DOC] Runs untrusted Python snippets in a child process. Holds a timeout in seconds, a
    # memory limit string, and a set of allowed top-level module names (math, random, datetime,
    # collections, itertools, functools, operator, statistics, decimal, fractions).
    # KNOWN ISSUE: the memory limit string is stored but never used; the child instead gets a
    # fixed 4 GB address-space limit written into the wrapper script.
    # KNOWN ISSUE: this is a pattern-matching screen, not a real sandbox; see the safety check.
    """Python code sandbox, provides safe code execution environment"""

    def __init__(self, timeout: int = 10, memory_limit: str = "100MB"):
        # [FUNCTION DOC] Store the timeout (default 10 seconds) and memory limit string (default
        # 100 megabytes, unused) and create the allowed module name set.
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.allowed_modules = {
            "math",
            "random",
            "datetime",
            "collections",
            "itertools",
            "functools",
            "operator",
            "statistics",
            "decimal",
            "fractions",
        }

    def _check_code_safety(self, code: str) -> tuple[bool, str]:
        # [FUNCTION DOC] HELPER: Screen code text. Returns a pair of safe flag and message. Step 1:
        # reject if any forbidden pattern is found anywhere in the text, case-insensitively:
        # imports of os, sys, subprocess, shutil, glob, pathlib; the dynamic import hook name;
        # calls to eval, exec, open, file, input, raw_input, compile, execfile, getattr, setattr,
        # delattr, hasattr, globals, locals, vars, dir, type, isinstance, issubclass, super,
        # property, staticmethod, classmethod; and any double-underscore name. The message names
        # the matched pattern. Step 2: collect the first module name after every import keyword and
        # every from keyword, and reject if any is not in the allowed set.
        # KNOWN ISSUE: matching is plain substring search with no word boundaries or parsing, so it
        # has false positives (for example a function whose name ends in dir, or the words from or
        # import inside comments or strings) and bypasses: a comma separated import list only has
        # its first module checked, so a forbidden module can follow an allowed one.
        """Check code safety by scanning for dangerous patterns"""
        # Check for dangerous operations
        dangerous_patterns = [
            r"import\s+os",
            r"import\s+sys",
            r"import\s+subprocess",
            r"import\s+shutil",
            r"import\s+glob",
            r"import\s+pathlib",
            r"__import__",
            r"eval\s*\(",
            r"exec\s*\(",
            r"open\s*\(",
            r"file\s*\(",
            r"input\s*\(",
            r"raw_input\s*\(",
            r"compile\s*\(",
            r"execfile\s*\(",
            r"getattr\s*\(",
            r"setattr\s*\(",
            r"delattr\s*\(",
            r"hasattr\s*\(",
            r"globals\s*\(",
            r"locals\s*\(",
            r"vars\s*\(",
            r"dir\s*\(",
            r"type\s*\(",
            r"isinstance\s*\(",
            r"issubclass\s*\(",
            r"super\s*\(",
            r"property\s*\(",
            r"staticmethod\s*\(",
            r"classmethod\s*\(",
            r"__\w+__",  # double underscore methods
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Code contains dangerous pattern: {pattern}"

        # Check imported modules
        import_pattern = r"import\s+(\w+)"
        from_pattern = r"from\s+(\w+)"

        imports = re.findall(import_pattern, code)
        froms = re.findall(from_pattern, code)

        all_imports = set(imports + froms)
        for imp in all_imports:
            if imp not in self.allowed_modules:
                return False, f"Import of '{imp}' is not allowed"

        return True, "Code is safe"

    @contextmanager
    def _create_safe_environment(self):
        # [FUNCTION DOC] HELPER: Context manager that makes a fresh temporary directory (name
        # prefix for the sandbox) and yields a triple: the path of a script file inside it (not yet
        # created), a copy of the current environment with the module path set to the directory and
        # unbuffered output on, and the directory. On exit removes the directory tree, ignoring
        # errors.
        """Create safe execution environment with temporary directory"""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="python_sandbox_")

        try:
            # Create safe Python script
            script_path = os.path.join(temp_dir, "code.py")

            # Set environment variables
            env = os.environ.copy()
            env["PYTHONPATH"] = temp_dir
            env["PYTHONUNBUFFERED"] = "1"

            yield script_path, env, temp_dir

        finally:
            # Clean up temporary directory
            try:
                import shutil

                shutil.rmtree(temp_dir)
            except Exception:
                pass

    async def execute_code(self, code: str) -> str:
        # [FUNCTION DOC] Run a code snippet and return a text result; failures are returned as text
        # starting with the word Error, never raised. Steps: if host process memory exceeds the
        # refusal limit, do aggressive cleanup and return an error asking to retry; run the safety
        # screen and return its message as an error if unsafe; wrap the code (every line indented
        # into a try block) in a script that sets a 4 GB address-space soft limit, redirects stdout
        # and stderr to buffers, and afterwards prints a result made of an output section and an
        # errors section, or, on an exception, an error message with traceback; write the script to
        # the temporary directory and run it with the system python3 as a child process whose
        # working directory is that directory, waiting up to the timeout. Exit code zero returns the
        # child's stripped standard output; nonzero returns an error with the code and the child's
        # standard error; timeout kills the child and returns a timeout error; any launch failure
        # returns an error with the message. Afterwards runs the memory check and prints a cleanup
        # message if it produced one.
        # KNOWN ISSUE: the child is started and awaited with blocking calls inside this coroutine,
        # so each execution blocks the event loop for up to the timeout and the concurrency
        # semaphore provides no real parallelism. Indenting every line also changes the contents of
        # multi-line string literals in the user code, and an exit request raised by the snippet
        # bypasses the capture restore so its output is lost.
        """Execute Python code in sandbox with safety checks"""
        # Check memory usage before execution
        current_memory = get_memory_usage()
        if current_memory > TOOL_CONFIGS["max_memory_usage"]:
            aggressive_cleanup_memory()
            return "Error: Memory usage too high, please try again"

        # Check code safety
        is_safe, message = self._check_code_safety(code)
        if not is_safe:
            return f"Error: {message}"

        # Add necessary wrapper code with memory limits
        # Properly indent the user code within the try block
        # Handle indentation properly by adding 4 spaces to each line
        indented_code = "\n".join("    " + line for line in code.split("\n"))

        wrapped_code = f"""import sys
import traceback
from io import StringIO
import resource

# Set memory limit (4GB)
try:
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024 * 1024 * 1024, -1))
except Exception:
    pass

# Redirect stdout and stderr
old_stdout = sys.stdout
old_stderr = sys.stderr
stdout_capture = StringIO()
stderr_capture = StringIO()
sys.stdout = stdout_capture
sys.stderr = stderr_capture

try:
    # User code
{indented_code}
    
    # Get output
    stdout_output = stdout_capture.getvalue()
    stderr_output = stderr_capture.getvalue()
    
    # Restore standard output
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    
    # Return result
    result = ""
    if stdout_output:
        result += f"Output:\\n{{stdout_output}}"
    if stderr_output:
        result += f"\\nErrors:\\n{{stderr_output}}"
    
    print(result)
    
except Exception as e:
    # Restore standard output
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    
    # Return error information
    error_msg = f"Error: {{str(e)}}\\nTraceback:\\n{{traceback.format_exc()}}"
    print(error_msg)"""

        with self._create_safe_environment() as (script_path, env, temp_dir):
            # Write code to file
            with open(script_path, "w") as f:
                f.write(wrapped_code)

            try:
                # Use subprocess to run code
                process = subprocess.Popen(
                    ["python3", script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    cwd=temp_dir,
                    text=True,
                )

                # Set timeout
                try:
                    stdout, stderr = process.communicate(timeout=self.timeout)

                    if process.returncode == 0:
                        result = stdout.strip()
                    else:
                        result = f"Error: Process exited with code {process.returncode}\n{stderr}"

                except subprocess.TimeoutExpired:
                    process.kill()
                    result = f"Error: Code execution timed out after {self.timeout} seconds"

            except Exception as e:
                result = f"Error: Failed to execute code: {str(e)}"

            # Check memory usage after execution and cleanup if needed
            cleanup_message = check_and_cleanup_memory()
            if cleanup_message:
                print(f"Memory cleanup: {cleanup_message}")

            return result


class ToolRegistry:
    # [CLASS DOC] Maps tool names to their function-calling specifications and dispatches
    # executions. Owns one Python sandbox configured from the module configuration (timeout and
    # memory limit string) and registers the code interpreter tool on construction.
    """Tool registry, manages available tools and their execution"""

    def __init__(self):
        # [FUNCTION DOC] Create an empty tool mapping, the sandbox (timeout and memory limit from
        # the module configuration) and register the default tools.
        self.tools = {}
        self.python_sandbox = PythonSandbox(
            timeout=TOOL_CONFIGS["python_timeout"], memory_limit=TOOL_CONFIGS["python_memory_limit"]
        )
        self._register_default_tools()

    def _register_default_tools(self):
        # [FUNCTION DOC] HELPER: Register the code interpreter tool: a function spec named code
        # interpreter with a description and an object parameter schema with one required string
        # property, the code to execute.
        """Register default tools in the registry"""
        # Python code interpreter
        self.register_tool(
            "code_interpreter",
            {
                "type": "function",
                "function": {
                    "name": "code_interpreter",
                    "description": "A tool for executing Python code in a safe sandbox environment.",
                    "parameters": {
                        "type": "object",
                        "properties": {"code": {"type": "string", "description": "The Python code to execute"}},
                        "required": ["code"],
                    },
                },
            },
        )

    def register_tool(self, name: str, tool_spec: dict[str, Any]):
        # [FUNCTION DOC] Store the spec under the name, replacing any existing entry. Registering
        # a spec does not add execution support; the dispatcher only implements the interpreter.
        """Register a new tool in the registry"""
        self.tools[name] = tool_spec

    def get_tool_specs(self) -> list[dict[str, Any]]:
        # [FUNCTION DOC] Return a new list of all registered specs in registration order.
        """Get all tool specifications as a list"""
        return list(self.tools.values())

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        # [FUNCTION DOC] Run a tool call and return text. An unknown name returns an error string
        # without waiting. Otherwise acquires the module concurrency semaphore; the interpreter name
        # runs the Python execution helper, and any other registered name returns a not-implemented
        # error string. Errors are returned, not raised.
        """Execute a tool call with the given arguments"""
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found"

        async with SEMAPHORE:
            if tool_name == "code_interpreter":
                return await self._execute_python(arguments)
            else:
                return f"Error: Tool '{tool_name}' not implemented"

    async def _execute_python(self, arguments: dict[str, Any]) -> str:
        # [FUNCTION DOC] HELPER: Take the code argument (empty string if absent); if it is a list,
        # join the items converted to text with newlines. Returns an error string for blank code;
        # otherwise returns the sandbox execution result.
        """Execute Python code using the sandbox"""
        code = arguments.get("code", "")
        if isinstance(code, list):
            code = "\n".join(str(item) for item in code)
        if not code.strip():
            return "Error: No code provided"

        # Execute code in sandbox
        result = await self.python_sandbox.execute_code(code)
        return result


tool_registry = ToolRegistry()

tool_specs = tool_registry.get_tool_specs()


async def execute_tool(name: str, params: dict) -> str:
    # [FUNCTION DOC] Module-level entry referenced from the launcher as the tool execution
    # function: delegates to the shared registry with the tool name and argument mapping and
    # returns its text result.
    return await tool_registry.execute_tool(name, params)


# Reward function that encourages tool usage
async def reward_func(args, sample: Sample, **kwargs):
    # [FUNCTION DOC] Custom reward function. The solution text is the prompt plus the response if
    # the prompt is a string, else just the response; the ground truth is the label or an empty
    # string; the tool call count comes from the sample metadata (which must be a mapping; zero if
    # absent). Scores with the DAPO math scorer requiring a boxed answer, giving a result mapping
    # with score, accuracy and prediction entries. If the score is negative (incorrect), adds a
    # bonus of 0.05 per tool call and then limits the score to at most -0.6, so tool use can lift an
    # incorrect answer's score only up to -0.6. A missing prediction is replaced by an empty
    # string. Returns the result mapping.
    # KNOWN ISSUE: the cap is applied with a minimum, so a negative base score already above -0.6
    # would be pushed down to -0.6 rather than left alone; this is only harmless if the scorer's
    # incorrect score is -1.
    """Tool call reward function using math_dapo, with bonus for tool usage."""
    solution_str = sample.prompt + sample.response if isinstance(sample.prompt, str) else sample.response
    ground_truth = sample.label if sample.label is not None else ""
    tool_call_count = sample.metadata.get("tool_call_count", 0)

    # use \boxed{...} answer
    result = math_dapo_compute_score(solution_str, ground_truth, strict_box_verify=True)

    # encourage model to call tools
    if result["score"] < 0:
        tool_call_reward = tool_call_count / 2 * 0.1
        result["score"] = min(-0.6, result["score"] + tool_call_reward)

    if result["pred"] is None:
        result["pred"] = ""

    return result
