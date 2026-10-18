# [MODULE DOC]
# Static hygiene checks over repository text files: shell launch scripts must not hard-code a
# checkout location, and docker patch files must not use removed helpers or removed shell model
# scripts. Constants: hard-coded checkout prefixes; regular expressions for a call to the removed
# exec-command helper (not preceded by a word character or dot), for a shell model script path under
# scripts/models, and for joining the result of the model args loader.
import re

from tests.fast.launch_scripts.sh_harness import REPO_ROOT

_HARDCODED_CHECKOUTS = ("/root/miles", "/workspace/miles")

_REMOVED_COMMAND_HELPERS = re.compile(r"(?<![\w.])exec_command\s*\(")

_REMOVED_MODEL_SCRIPTS = re.compile(r"scripts/models/\S+\.sh")

_JOINED_MODEL_ARGS = re.compile(r"join\(\s*load_model_args")


class TestShellScriptHygiene:
    # [CLASS DOC] Hygiene checks over shell scripts under the scripts and examples directories.
    def test_no_shell_script_hardcodes_the_checkout_location(self):
        # [FUNCTION DOC] Lists repo-relative paths of shell files whose text contains either hard-coded
        # checkout prefix (undecodable bytes replaced) and asserts the list is empty.
        """A script that assumes one absolute checkout only runs inside one container image."""
        offenders = [
            path.relative_to(REPO_ROOT).as_posix()
            for root in (REPO_ROOT / "scripts", REPO_ROOT / "examples")
            for path in root.rglob("*.sh")
            for text in [path.read_text(errors="replace")]
            if any(hardcoded in text for hardcoded in _HARDCODED_CHECKOUTS)
        ]

        assert offenders == []


class TestDockerPatchHygiene:
    # [CLASS DOC] Hygiene checks over every patch file under the docker directory.
    def test_no_patch_calls_the_removed_exec_command_helper(self):
        # [FUNCTION DOC] Asserts no patch file text matches the removed exec-command call pattern.
        """A patch is applied at image build time, so a stale symbol only fails on the target hardware."""
        offenders = [
            path.relative_to(REPO_ROOT).as_posix()
            for path in (REPO_ROOT / "docker").rglob("*.patch")
            if _REMOVED_COMMAND_HELPERS.search(path.read_text(errors="replace"))
        ]

        assert offenders == []

    def test_no_patch_joins_the_model_args_line(self):
        # [FUNCTION DOC] Asserts no patch file text matches the pattern of joining the model args
        # loader's result.
        """load_model_args() returns the line; joining it would splice the string character by character."""
        offenders = [
            path.relative_to(REPO_ROOT).as_posix()
            for path in (REPO_ROOT / "docker").rglob("*.patch")
            if _JOINED_MODEL_ARGS.search(path.read_text(errors="replace"))
        ]

        assert offenders == []

    def test_no_patch_sources_a_removed_shell_model_script(self):
        # [FUNCTION DOC] Asserts no patch file text mentions a shell model script path under the
        # scripts models directory.
        """The model definitions are python now, so a `source scripts/models/x.sh` can never resolve."""
        offenders = [
            path.relative_to(REPO_ROOT).as_posix()
            for path in (REPO_ROOT / "docker").rglob("*.patch")
            if _REMOVED_MODEL_SCRIPTS.search(path.read_text(errors="replace"))
        ]

        assert offenders == []
