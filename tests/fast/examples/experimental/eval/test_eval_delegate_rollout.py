# [MODULE DOC]
# Tests for how the eval delegate rollout example module obtains its delegate client from the
# evaluation config argument. The config value may be an inline pseudo-file (a reserved prefix
# followed by base64 of the yaml text) or a plain file path. The delegate-config rebuilder is
# replaced by an identity on the entries and the client factory is replaced by a recorder, so each
# test observes the list of delegate entries that would be used to create the client.
import base64
from types import SimpleNamespace

import pytest

from examples.experimental.eval import eval_delegate_rollout
from miles.utils.file_arg_utils import PSEUDO_FILE_PREFIX

_CONFIG = """
eval:
  delegate:
    - name: aime
"""


@pytest.fixture
def recorded_env_configs(monkeypatch):
    # [FUNCTION DOC] HELPER: Fixture that patches the module's delegate-config rebuilder to return
    # its entries unchanged and the delegate client's maybe-create class method to append the env
    # configs it receives to a list (and return nothing). Clears the module's delegate cache.
    # Returns the live recording list.
    seen = []
    monkeypatch.setattr(eval_delegate_rollout, "_rebuild_delegate_config", lambda args, entries, defaults: entries)
    monkeypatch.setattr(
        eval_delegate_rollout.EvalDelegateClient,
        "maybe_create",
        classmethod(lambda cls, args, env_configs: seen.append(env_configs)),
    )
    eval_delegate_rollout._DELEGATE_CACHE.clear()
    return seen


class TestGetDelegateClient:
    # [CLASS DOC] Tests of the delegate client getter's handling of the eval config argument.
    def test_accepts_an_inline_eval_config(self, recorded_env_configs):
        # [FUNCTION DOC] Encodes the sample yaml as base64 behind the pseudo-file prefix, calls the
        # getter, and asserts the recorded env configs are exactly one list with the one aime entry.
        """The main parser resolves --eval-config, so the delegate must resolve the same value too."""
        encoded = base64.b64encode(_CONFIG.encode()).decode()
        args = SimpleNamespace(eval_config=f"{PSEUDO_FILE_PREFIX}{encoded}")

        eval_delegate_rollout._get_delegate_client(args)

        assert recorded_env_configs == [[{"name": "aime"}]]

    def test_accepts_a_plain_eval_config_path(self, recorded_env_configs, tmp_path):
        # [FUNCTION DOC] Writes the sample yaml to a temp file, calls the getter with its path, and
        # asserts the same recorded result as for the inline form.
        """A file path keeps working and is still cached by mtime."""
        path = tmp_path / "eval.yaml"
        path.write_text(_CONFIG)
        args = SimpleNamespace(eval_config=str(path))

        eval_delegate_rollout._get_delegate_client(args)

        assert recorded_env_configs == [[{"name": "aime"}]]
