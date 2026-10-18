# [MODULE DOC]
# Tests of the shell launch-script harness itself: running a real launcher (the qwen3 4b fp8 example)
# through shims, ordering and completeness of recorded background commands, rejection of environment
# overrides that would unfreeze the run, and shim edge cases (gpu wait loops, intercepted downloads
# and torchrun, repo path placeholders). Constants: the real script path and four inline bash scripts:
# one backgrounding a server, one backgrounding inside a command substitution, and one synthetic
# script with a gpu wait loop, a model download, a torchrun call and a ray job submit whose checkout
# placeholder is replaced by the repo root.
import pytest

from tests.fast.launch_scripts import sh_harness
from tests.fast.launch_scripts.sh_harness import REPO_ROOT, REPO_ROOT_PLACEHOLDER, run_launch_script

_SCRIPT = REPO_ROOT / "examples" / "infra_features" / "low_precision" / "run-qwen3-4b-fp8.sh"

_BACKGROUNDING_SCRIPT = """#!/bin/bash
set -ex
python3 -m sglang.launch_server --port 13141 >/dev/null 2>&1 &
curl -sf http://127.0.0.1:13141/health_generate
ray job submit --address="http://127.0.0.1:8265" -- python3 train.py
"""

_BACKGROUNDING_INSIDE_A_SUBSTITUTION_SCRIPT = """#!/bin/bash
set -ex
start_server() {
    python3 -m sglang.launch_server --port "$1" >/dev/null 2>&1 &
    echo "/tmp/server-$1.log"
}
LOG=$(start_server 13141)
curl -sf http://127.0.0.1:13141/health_generate
ray job submit --address="http://127.0.0.1:8265" -- python3 train.py "$LOG"
"""

_SYNTHETIC_SCRIPT = """#!/bin/bash
set -ex
EXPECTED_GPUS=32
while true; do
    AVAILABLE_GPUS=$(python3 -c "import ray; print(int(ray.cluster_resources().get('GPU', 0)))" 2>/dev/null || echo 0)
    if [ "$AVAILABLE_GPUS" -ge "$EXPECTED_GPUS" ]; then
        break
    fi
    sleep 5
done
hf download some/model --local-dir /root/models/some-model
torchrun --nproc-per-node 8 CHECKOUT/tools/convert_hf_to_torch_dist.py
ray job submit --address="http://127.0.0.1:8265" -- python3 CHECKOUT/train.py
"""


class TestRunLaunchScriptOnARealScript:
    # [CLASS DOC] Tests running the real qwen3 4b fp8 example launcher under the harness; a fixture runs
    # it once per test in the temp dir.
    @pytest.fixture
    def run(self, tmp_path):
        # [FUNCTION DOC] HELPER: Fixture. Runs the real script with the temp dir as sandbox and returns
        # the run result.
        return run_launch_script(_SCRIPT, sandbox=tmp_path)

    def test_script_runs_to_completion_without_touching_the_real_system(self, run):
        # [FUNCTION DOC] Asserts the shell's return code is zero.
        """The shimmed PATH lets a real launch script run end to end and exit cleanly."""
        assert run.returncode == 0

    def test_destructive_commands_are_intercepted_instead_of_executed(self, run):
        # [FUNCTION DOC] Asserts the recorded invocations include the force-kill of sglang processes
        # and the forced ray stop, exactly as argument lists.
        """pkill / ray stop are recorded by shims, so they never reach the test runner."""
        assert ["pkill", "-9", "sglang"] in run.invocations
        assert ["ray", "stop", "--force"] in run.invocations

    def test_ray_start_is_recorded_with_the_frozen_master_addr(self, run):
        # [FUNCTION DOC] Expects exactly one recorded ray start invocation and asserts the argument after
        # its node ip address flag is the frozen master address 127.0.0.1.
        """Node address comes from the frozen environment, not from the developer machine."""
        (ray_start,) = [argv for argv in run.invocations_of("ray") if argv[1] == "start"]
        assert "--node-ip-address" in ray_start
        assert ray_start[ray_start.index("--node-ip-address") + 1] == "127.0.0.1"

    def test_ray_job_submit_argv_contains_the_expanded_model_args(self, run):
        # [FUNCTION DOC] Takes the unique ray job submit invocation; asserts it begins with ray, job,
        # submit, the num layers flag is followed by 36, and the hf checkpoint flag is followed by the
        # root-level Qwen3 4B FP8 path.
        """The scripts/models/*.py expansion must be visible in the captured argv."""
        argv = run.ray_job_submit_argv()
        assert argv[:3] == ["ray", "job", "submit"]
        assert "--num-layers" in argv
        assert argv[argv.index("--num-layers") + 1] == "36"
        assert argv[argv.index("--hf-checkpoint") + 1] == "/root/Qwen3-4B-FP8"

    def test_nvlink_detection_is_frozen_to_absent(self, run):
        # [FUNCTION DOC] Finds the single runtime-environment json argument of the submit and asserts it
        # sets the nccl nvls enable variable to the string 0.
        """The nvidia-smi shim reports no NVLink, so NCCL_NVLS_ENABLE is deterministic."""
        argv = run.ray_job_submit_argv()
        (runtime_env,) = [arg for arg in argv if arg.startswith("--runtime-env-json=")]
        assert '"NCCL_NVLS_ENABLE": "0"' in runtime_env

    def test_reruns_produce_identical_recordings(self, tmp_path):
        # [FUNCTION DOC] Runs the real script twice in two different sandboxes and asserts the
        # sanitized invocation lists are equal.
        """Snapshot testing only works if the harness is deterministic across runs."""
        first = run_launch_script(_SCRIPT, sandbox=tmp_path / "a")
        second = run_launch_script(_SCRIPT, sandbox=tmp_path / "b")

        assert first.invocations == second.invocations


class TestRunLaunchScriptOnABackgroundingScript:
    # [CLASS DOC] Tests with inline scripts that start a background process, checking that records
    # from background children are all captured and ordered by pid.
    @pytest.fixture
    def script(self, tmp_path):
        # [FUNCTION DOC] HELPER: Fixture. Writes the backgrounding script text to a file in the temp dir
        # and returns its path.
        script = tmp_path / "backgrounding.sh"
        script.write_text(_BACKGROUNDING_SCRIPT)
        return script

    def test_a_backgrounded_command_is_still_recorded(self, script, tmp_path):
        # [FUNCTION DOC] Runs the script (30 s timeout) and asserts a zero return code and that the first
        # python3 record begins with the module flag and the sglang launch server module.
        """bash exits without reaping `&` children, so reading the capture too early loses them."""
        run = run_launch_script(script, sandbox=tmp_path / "sandbox", timeout=30)

        assert run.returncode == 0
        assert run.invocations_of("python3")[0][1:3] == ["-m", "sglang.launch_server"]

    @pytest.mark.parametrize("attempt", range(20))
    def test_backgrounding_does_not_perturb_the_recorded_order(self, script, tmp_path, attempt):
        # [FUNCTION DOC] Repeated for 20 attempts, each in a fresh sandbox. Asserts the recorded command
        # names are exactly python3, curl, ray in that order.
        """Snapshots assert an exact sequence, so a `&` must not shuffle records run to run."""
        run = run_launch_script(script, sandbox=tmp_path / f"sandbox-{attempt}", timeout=30)

        assert [argv[0] for argv in run.invocations] == ["python3", "curl", "ray"]

    def test_records_are_ordered_by_the_fork_that_made_them_not_by_arrival(self, tmp_path):
        # [FUNCTION DOC] Unit test of the private capture parser: feeds a raw capture where the record
        # with pid 12 (curl) precedes the record with pid 11 (python3), and asserts the parsed result is
        # sorted by pid, giving python3 then curl (each as a one-element list).
        """bash forks in command order, so the pid orders records even when a `&` child appends late."""
        late_background = f"11{sh_harness._ARG_SEPARATOR}python3{sh_harness._RECORD_SEPARATOR}"
        foreground = f"12{sh_harness._ARG_SEPARATOR}curl{sh_harness._RECORD_SEPARATOR}"

        parsed = sh_harness._parse_capture(foreground + late_background, sandbox=tmp_path)

        assert parsed == [["python3"], ["curl"]]

    def test_a_command_backgrounded_inside_a_substitution_does_not_hang_the_run(self, tmp_path):
        # [FUNCTION DOC] Writes and runs the script that backgrounds a command inside a command
        # substitution (30 s timeout); asserts a zero return code (no hang on zombie processes) and the
        # command order python3, curl, ray.
        """Orphaned by its subshell, it lingers as a zombie wherever PID 1 does not reap, and killpg still sees the group."""
        script = tmp_path / "substitution.sh"
        script.write_text(_BACKGROUNDING_INSIDE_A_SUBSTITUTION_SCRIPT)

        run = run_launch_script(script, sandbox=tmp_path / "sandbox", timeout=30)

        assert run.returncode == 0
        assert [argv[0] for argv in run.invocations] == ["python3", "curl", "ray"]


class TestRunLaunchScriptEnvironmentFreeze:
    # [CLASS DOC] Tests of the extra environment argument and the frozen variable guard.
    def test_extra_env_may_not_shadow_a_frozen_variable(self, tmp_path):
        # [FUNCTION DOC] Passing a different master address expects an assertion error naming it.
        """A caller overriding MASTER_ADDR would unfreeze every snapshot that records it."""
        with pytest.raises(AssertionError, match="MASTER_ADDR"):
            run_launch_script(_SCRIPT, sandbox=tmp_path, extra_env={"MASTER_ADDR": "10.0.0.9"})

    def test_extra_env_may_not_shadow_the_capture_channel(self, tmp_path):
        # [FUNCTION DOC] Passing the harness capture-path variable expects an assertion error naming it.
        """Redirecting the capture path would make every shim record vanish silently."""
        with pytest.raises(AssertionError, match="MILES_SH_HARNESS_CAPTURE"):
            run_launch_script(_SCRIPT, sandbox=tmp_path, extra_env={"MILES_SH_HARNESS_CAPTURE": "/dev/null"})

    def test_extra_env_may_still_supply_a_variable_the_harness_does_not_own(self, tmp_path):
        # [FUNCTION DOC] Runs the real script with an unrelated base folder variable and asserts a zero
        # return code.
        """The freeze must not block the per-script inputs the snapshot suite has to pass in."""
        run = run_launch_script(_SCRIPT, sandbox=tmp_path, extra_env={"BASE_FOLDER": "/frozen/checkpoints"})

        assert run.returncode == 0


class TestRunLaunchScriptOnTheShimEdgeCases:
    # [CLASS DOC] Tests of shim behavior using a synthetic script run once per test by a fixture.
    @pytest.fixture
    def run(self, tmp_path):
        # [FUNCTION DOC] HELPER: Fixture. Writes the synthetic script with the checkout placeholder
        # replaced by the repo root, runs it with a 30 s timeout in a sandbox subdirectory, and returns
        # the result.
        script = tmp_path / "synthetic.sh"
        script.write_text(_SYNTHETIC_SCRIPT.replace("CHECKOUT", str(REPO_ROOT)))
        return run_launch_script(script, sandbox=tmp_path / "sandbox", timeout=30)

    def test_a_gpu_wait_loop_leaves_on_its_first_poll(self, run):
        # [FUNCTION DOC] Asserts a zero return code and that sleep was never invoked, i.e. the python
        # shim reported enough gpus on the first poll.
        """The python shim must emit a real number; emitting its repr spins the loop until timeout."""
        assert run.returncode == 0
        assert run.invocations_of("sleep") == []

    def test_downloads_and_torchrun_are_intercepted(self, run):
        # [FUNCTION DOC] Asserts the first hf record's second argument is download and the first
        # torchrun record's second argument is the nproc-per-node flag.
        """Unshimmed, these would pull real weights and start a real training job."""
        assert run.invocations_of("hf")[0][1] == "download"
        assert run.invocations_of("torchrun")[0][1] == "--nproc-per-node"

    def test_a_repo_path_inside_argv_becomes_a_placeholder(self, run):
        # [FUNCTION DOC] Asserts the last argument of the torchrun record is the conversion tool path
        # prefixed with the repo placeholder, and the real repo root appears nowhere in the record.
        """Recordings must not embed the checkout location of whoever ran the test."""
        torchrun_argv = run.invocations_of("torchrun")[0]

        assert torchrun_argv[-1] == f"{REPO_ROOT_PLACEHOLDER}/tools/convert_hf_to_torch_dist.py"
        assert str(REPO_ROOT) not in " ".join(torchrun_argv)
