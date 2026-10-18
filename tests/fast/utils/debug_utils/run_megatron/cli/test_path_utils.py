# [MODULE DOC] Tests for path helpers of the run_megatron tool
# (miles.utils.debug_utils.run_megatron.cli.path_utils): resolving the Megatron checkout path
# (explicit argument wins, then the MEGATRON_PATH environment variable, then the default
# "/root/Megatron-LM") and resolving a model script under scripts/models of the repository base
# (raises a typer bad-parameter error when the script file is missing).
from pathlib import Path

import pytest
import typer

from miles.utils.debug_utils.run_megatron.cli.path_utils import resolve_megatron_path, resolve_model_script


class TestResolveMegatronPath:
    # [CLASS DOC] Tests of the Megatron path resolution order.
    def test_explicit_path(self) -> None:
        # [FUNCTION DOC] An explicit path is returned as the very same object.
        explicit = Path("/custom/megatron")
        assert resolve_megatron_path(explicit) is explicit

    def test_env_var_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] With no explicit path and the environment variable set to "/env/megatron",
        # the result equals that path.
        monkeypatch.setenv("MEGATRON_PATH", "/env/megatron")
        assert resolve_megatron_path(None) == Path("/env/megatron")

    def test_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] With no explicit path and the environment variable removed, the result is
        # "/root/Megatron-LM".
        monkeypatch.delenv("MEGATRON_PATH", raising=False)
        assert resolve_megatron_path(None) == Path("/root/Megatron-LM")

    def test_explicit_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] With both an explicit path and the environment variable set, the explicit
        # path object is returned.
        monkeypatch.setenv("MEGATRON_PATH", "/env/megatron")
        explicit = Path("/custom/megatron")
        assert resolve_megatron_path(explicit) is explicit


class TestResolveModelScript:
    # [CLASS DOC] Tests of locating a model script file; the repository base lookup function is
    # patched to return the temporary directory.
    def test_returns_path_when_exists(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Creates an empty deepseek_v3 script file in scripts/models under the
        # temporary base and asserts resolving "deepseek_v3" returns that file's path.
        scripts_dir = tmp_path / "scripts" / "models"
        scripts_dir.mkdir(parents=True)
        script_file = scripts_dir / "deepseek_v3.py"
        script_file.touch()

        monkeypatch.setattr(
            "miles.utils.debug_utils.run_megatron.cli.path_utils._resolve_repo_base",
            lambda: tmp_path,
        )
        result = resolve_model_script("deepseek_v3")
        assert result == script_file

    def test_raises_when_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] With an empty scripts/models directory, resolving "nonexistent" raises a
        # typer bad-parameter error whose message says "Model script not found".
        scripts_dir = tmp_path / "scripts" / "models"
        scripts_dir.mkdir(parents=True)

        monkeypatch.setattr(
            "miles.utils.debug_utils.run_megatron.cli.path_utils._resolve_repo_base",
            lambda: tmp_path,
        )
        with pytest.raises(typer.BadParameter, match="Model script not found"):
            resolve_model_script("nonexistent")
