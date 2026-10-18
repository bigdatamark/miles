# [MODULE DOC] Tests for the helm environment helpers: current_namespace (the namespace
# environment variable wins; otherwise the service account namespace file, stripped; a missing or
# whitespace-only file raises an assertion error naming the variable or saying "is empty"),
# current_release (read from its environment variable; missing raises an assertion error naming
# the variable) and the rule that the per-pod base gpu id annotation key must not start with the
# meta annotation prefix. Environment variables and the namespace file path are patched through
# the monkeypatch.
from __future__ import annotations

from pathlib import Path

import pytest

from miles.utils.workers.worker_provider.kubernetes.helm import env


class TestNamespaceDiscovery:
    # [CLASS DOC] Tests of namespace discovery order and failure modes.
    def test_prefers_the_namespace_the_environment_names(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] With the namespace variable set to team-b the result is team-b.
        """A driver run outside a pod has no service account file to read."""
        monkeypatch.setenv(env.NAMESPACE_ENV_VAR, "team-b")

        assert env.current_namespace() == "team-b"

    def test_reads_the_namespace_of_its_own_pod(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        # [FUNCTION DOC] Variable unset; the patched namespace file holds "team-c" with a trailing
        # newline; the result is "team-c" (whitespace stripped).
        """In a pod nobody passes the namespace in, but the service account always says it."""
        monkeypatch.delenv(env.NAMESPACE_ENV_VAR, raising=False)
        namespace_file = tmp_path / "namespace"
        namespace_file.write_text("team-c\n")
        monkeypatch.setattr(env, "NAMESPACE_FILE", namespace_file)

        assert env.current_namespace() == "team-c"

    def test_refuses_to_guess(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        # [FUNCTION DOC] Variable unset and the namespace file missing: an assertion error whose
        # message names the variable.
        """Guessing 'default' would watch someone else's pods and heal them."""
        monkeypatch.delenv(env.NAMESPACE_ENV_VAR, raising=False)
        monkeypatch.setattr(env, "NAMESPACE_FILE", tmp_path / "missing")

        with pytest.raises(AssertionError, match=env.NAMESPACE_ENV_VAR):
            env.current_namespace()

    def test_refuses_an_empty_namespace_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        # [FUNCTION DOC] Variable unset and the file holds only whitespace: an assertion error
        # saying "is empty".
        """A whitespace-only service account file cannot identify a namespace."""
        monkeypatch.delenv(env.NAMESPACE_ENV_VAR, raising=False)
        namespace_file = tmp_path / "namespace"
        namespace_file.write_text(" \n\t")
        monkeypatch.setattr(env, "NAMESPACE_FILE", namespace_file)

        with pytest.raises(AssertionError, match="is empty"):
            env.current_namespace()


class TestReleaseDiscovery:
    # [CLASS DOC] Tests of release name discovery.
    def test_reads_the_release_the_chart_told_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] With the release variable set, the release is that value.
        """The orchestrator cannot recompute the release name, because its run uuid is its own."""
        monkeypatch.setenv(env.RELEASE_ENV_VAR, "miles-run-260805")

        assert env.current_release() == "miles-run-260805"

    def test_refuses_to_guess_the_release(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] With the variable unset, an assertion error naming the variable.
        """A guessed release selects no pods at all, and the run would wait forever for its cells."""
        monkeypatch.delenv(env.RELEASE_ENV_VAR, raising=False)

        with pytest.raises(AssertionError, match=env.RELEASE_ENV_VAR):
            env.current_release()


class TestTheCardAnnotationStandsApartFromTheMeta:
    # [CLASS DOC] Checks the default label key set for annotation key separation.
    def test_the_base_gpu_id_key_is_not_a_meta_annotation(self) -> None:
        # [FUNCTION DOC] Asserts the default base gpu id annotation key does not start with the
        # default meta annotation prefix.
        """Meta annotations are merged across a cell's pods and asserted equal; this one differs per pod."""
        keys = env.DEFAULT_LABEL_KEYS

        assert not keys.base_gpu_id_annotation.startswith(keys.meta_annotation_prefix)
