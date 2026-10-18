# [MODULE DOC] Manual tests of the model argument tables used by the launch scripts. For every
# model type discovered by the shared harness, the expanded argument tokens are compared with a
# golden snapshot file (one JSON-encoded token per line) under the snapshots directory, and are
# checked to be flags and values without whitespace; a discovery test asserts the discovered model
# types equal the snapshot file stems and number more than 60.
import json

import pytest

from tests.fast.launch_scripts.model_args_harness import expand_model_args, iter_model_types
from tests.fast.launch_scripts.sh_harness import REPO_ROOT, assert_matches_snapshot

_SNAPSHOT_DIR = REPO_ROOT / "tests" / "snapshots" / "model_args"

_MODEL_TYPES = iter_model_types()


class TestEveryModelType:
    # [CLASS DOC] Per-model checks, parametrized over every discovered model type.
    @pytest.mark.parametrize("model_type", _MODEL_TYPES)
    def test_model_args_match_snapshot(self, model_type: str) -> None:
        # [FUNCTION DOC] Renders the expanded tokens as one JSON string per line with a trailing
        # newline and asserts it matches the model's snapshot file (the harness decides how a
        # mismatch or a missing snapshot is handled).
        """The golden argv of every model, so a later rewrite of the model definitions cannot drift."""
        actual = "\n".join(json.dumps(token) for token in expand_model_args(model_type)) + "\n"

        assert_matches_snapshot(_SNAPSHOT_DIR / f"{model_type}.txt", actual, model_type)

    @pytest.mark.parametrize("model_type", _MODEL_TYPES)
    def test_model_args_are_flags_and_values(self, model_type: str) -> None:
        # [FUNCTION DOC] The token list is non-empty, starts with a token beginning with two
        # dashes, and no token has leading or trailing whitespace or contains a space.
        """Consumers split the args on whitespace, so a token that contains any would silently become two."""
        tokens = expand_model_args(model_type)

        assert tokens
        assert tokens[0].startswith("--")
        assert all(token == token.strip() and " " not in token for token in tokens)


class TestDiscovery:
    # [CLASS DOC] Checks that discovery and snapshots stay in sync.
    def test_every_model_is_discovered_and_snapshotted(self) -> None:
        # [FUNCTION DOC] The set of discovered model types equals the set of snapshot file stems,
        # and more than 60 model types exist.
        """A model that stops matching the discovery glob would otherwise lose its golden file silently."""
        snapshotted = {path.stem for path in _SNAPSHOT_DIR.glob("*.txt")}

        assert set(_MODEL_TYPES) == snapshotted
        assert len(_MODEL_TYPES) > 60
