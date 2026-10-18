# [MODULE DOC] Tests of the agentic sandbox-provider credential helpers: the per-backend
# credential spec table, launcher-side key supply (credential files are forwarded by path, never by
# value; environment supply is accepted without forwarding; multi-variable providers need every
# variable), safe forwarding of endpoint addresses, SDK import and minimum-version preflight,
# worker-side key resolution (environment first, then a file, then a default path), and whole
# provider provisioning. Environment variables are manipulated through the monkeypatch fixture and
# files live in pytest temporary directories; no network or real provider SDK is used.
import sys
import types

import pytest

import miles.rollout.agentic.credentials as credentials
from miles.rollout.agentic.credentials import (
    PROVIDER_CREDENTIALS,
    credential_available,
    forward_address,
    preflight_sdk,
    provision_provider,
    resolve_provider_api_key,
    sandbox_key_supply,
)

_SPEC_KEYS = {
    "provider",
    "key_env_vars",
    "file_env_var",
    "arg_attr",
    "default_path",
    "provision_hint",
    "sdk",
    "sdk_hint",
    "forward",
    "target",
}
# Per-backend extras a spec may carry; the launchers read them with .get().
_OPTIONAL_SPEC_KEYS = {"env_defaults", "sdk_min_version"}


# --- the credential table ----------------------------------------------------


@pytest.mark.parametrize("backend", sorted(PROVIDER_CREDENTIALS))
def test_credential_spec_is_complete(backend):
    # [FUNCTION DOC] Parametrized over every backend in the credential table. Asserts the spec's
    # key set contains all required keys and no keys beyond the required plus optional ones, and
    # that the key environment variable list and the default path are non-empty.
    spec = PROVIDER_CREDENTIALS[backend]
    assert _SPEC_KEYS <= set(spec) <= _SPEC_KEYS | _OPTIONAL_SPEC_KEYS, backend
    assert spec["key_env_vars"], backend
    assert spec["default_path"], backend


def test_forwarded_vars_are_addresses_not_secrets():
    # [FUNCTION DOC] For every backend, asserts that none of the variable names the spec forwards
    # contains the words key, token or secret (upper case), since forwarded values are logged.
    """Whatever `forward` names is forwarded BY VALUE through ray's runtime_env,
    which is logged in plaintext — so no credential-shaped var may appear."""
    for backend, spec in PROVIDER_CREDENTIALS.items():
        for var in spec["forward"]:
            assert not any(word in var for word in ("KEY", "TOKEN", "SECRET")), (backend, var)


def test_a_forwarded_url_may_not_smuggle_a_credential():
    # [FUNCTION DOC] Forwards a plain endpoint URL into an environment mapping and asserts it is
    # stored under its variable name; then forwards a URL with user and password in its authority
    # part and asserts a value error saying it embeds credentials.
    """The check above is on the var's NAME, which cannot see a credential
    hidden in an endpoint's userinfo — and that value would be forwarded and
    logged in plaintext just the same."""
    env: dict[str, str] = {}
    forward_address(env, "E2B_API_URL", "https://agentenv.internal")
    assert env == {"E2B_API_URL": "https://agentenv.internal"}

    with pytest.raises(ValueError, match="embeds credentials"):
        forward_address({}, "E2B_API_URL", "https://user:tok@agentenv.internal")


# --- launcher-side key supply -------------------------------------------------


def _supply(env, spec_name, *, arg_path="", **overrides):
    # [FUNCTION DOC] HELPER: Calls the sandbox key supply routine on the given environment mapping
    # using the named backend's spec values (provider, key variables, file variable, default path,
    # provision hint) and the given explicit key path (default empty). Keyword overrides replace any
    # of those argument values. The supply routine mutates the mapping and may raise a value error.
    spec = PROVIDER_CREDENTIALS[spec_name]
    kwargs = {
        "provider": spec["provider"],
        "key_env_vars": spec["key_env_vars"],
        "file_env_var": spec["file_env_var"],
        "arg_path": arg_path,
        "default_path": spec["default_path"],
        "provision_hint": spec["provision_hint"],
    }
    sandbox_key_supply(env, **{**kwargs, **overrides})


def test_readable_file_forwards_the_path_never_the_value(tmp_path):
    # [FUNCTION DOC] Writes a key file, supplies it for the daytona backend, and asserts the
    # environment mapping holds only the file-path variable pointing at the file and does not
    # contain the secret value.
    key_file = tmp_path / "api_key"
    key_file.write_text("dtn_secret_value\n")
    env: dict[str, str] = {}
    _supply(env, "daytona", arg_path=str(key_file))
    assert env == {"DAYTONA_API_KEY_FILE": str(key_file)}
    assert "dtn_secret_value" not in str(env)


def test_configured_path_that_does_not_resolve_is_an_error(tmp_path):
    # [FUNCTION DOC] Supplies an explicit key path that does not exist for the e2b backend and
    # asserts a value error saying it is missing or empty.
    with pytest.raises(ValueError, match="is missing or empty"):
        _supply({}, "e2b", arg_path=str(tmp_path / "absent"))


def test_empty_file_is_not_a_credential(tmp_path):
    # [FUNCTION DOC] Supplies an explicit key path whose file holds only whitespace and asserts a
    # value error saying it is missing or empty.
    key_file = tmp_path / "api_key"
    key_file.write_text("   \n")
    with pytest.raises(ValueError, match="is missing or empty"):
        _supply({}, "e2b", arg_path=str(key_file))


def test_unreadable_path_counts_as_absent(monkeypatch, tmp_path):
    # [FUNCTION DOC] Sets the e2b key in the process environment and makes the default path a
    # directory (reading it fails). Asserts supply still succeeds and forwards nothing.
    """A path that raises on read (here: a directory) is 'no file', so the env
    supply still satisfies — the launcher cannot probe worker nodes anyway."""
    monkeypatch.setenv("E2B_API_KEY", "e2b_from_env")
    env: dict[str, str] = {}
    _supply(env, "e2b", default_path=str(tmp_path))
    assert env == {}


def test_worker_environment_supply_is_accepted_without_forwarding(monkeypatch, tmp_path):
    # [FUNCTION DOC] Sets the daytona key in the environment with a nonexistent default path and
    # asserts supply succeeds and forwards nothing.
    """When the launcher itself has the credential in env, workers are assumed
    to have it too (platform-injected / single-host inheritance) and nothing is
    forwarded."""
    monkeypatch.setenv("DAYTONA_API_KEY", "dtn_from_env")
    env: dict[str, str] = {}
    _supply(env, "daytona", default_path=str(tmp_path / "absent"))
    assert env == {}


def test_modal_token_pair_must_be_complete(monkeypatch, tmp_path):
    # [FUNCTION DOC] With only the modal token id set (secret removed) and no file, asserts a value
    # error naming both token variables. After also setting the secret, asserts supply succeeds
    # and forwards nothing (the token halves are never forwarded).
    """Half a token pair is a misconfiguration, not a usable credential: it
    would authenticate nothing and fail every episode."""
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-123")
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    with pytest.raises(ValueError, match="MODAL_TOKEN_ID \\+ MODAL_TOKEN_SECRET"):
        _supply({}, "modal", default_path=str(tmp_path / "absent"))

    monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-456")
    env: dict[str, str] = {}
    _supply(env, "modal", default_path=str(tmp_path / "absent"))
    assert env == {}  # the token halves are never forwarded


def test_modal_config_file_is_forwarded_by_path(tmp_path):
    # [FUNCTION DOC] Writes a modal config file containing a token id and secret, supplies it, and
    # asserts the environment holds only the modal config path variable pointing at the file and
    # does not contain the secret.
    """Modal's file is a config file rather than a bare key, but it rides the
    same path-not-value contract."""
    config = tmp_path / "modal.toml"
    config.write_text('[radixark]\ntoken_id = "ak-123"\ntoken_secret = "as-456"\n')
    env: dict[str, str] = {}
    _supply(env, "modal", arg_path=str(config))
    assert env == {"MODAL_CONFIG_PATH": str(config)}
    assert "as-456" not in str(env)


def test_missing_credential_names_what_to_provision(monkeypatch, tmp_path):
    # [FUNCTION DOC] With no e2b key in the environment and a nonexistent default path, asserts the
    # raised value error includes the provisioning hint (a command creating the e2b config
    # directory).
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    with pytest.raises(ValueError, match="mkdir -p ~/.config/e2b"):
        _supply({}, "e2b", default_path=str(tmp_path / "absent"))


# --- SDK preflight -------------------------------------------------------------


def test_preflight_passes_when_the_sdk_imports(monkeypatch):
    # [FUNCTION DOC] Registers an empty fake module under a fake SDK name and asserts the SDK
    # preflight returns without raising.
    monkeypatch.setitem(sys.modules, "fake_provider_sdk", types.ModuleType("fake_provider_sdk"))
    preflight_sdk("fake_provider_sdk", "pip install fake-provider-sdk")


def test_preflight_names_the_install_hint_when_the_sdk_is_missing():
    # [FUNCTION DOC] Runs preflight for a module name that cannot be imported and asserts a
    # runtime error that includes the given install hint.
    with pytest.raises(RuntimeError, match="pip install nonexistent-sdk"):
        preflight_sdk("nonexistent_provider_sdk", "pip install nonexistent-sdk")


def test_preflight_enforces_min_version(monkeypatch):
    # [FUNCTION DOC] Fakes the package metadata version lookup. With installed version 2.11.0 and
    # a minimum of 2.12, asserts a runtime error stating the requirement and installed version.
    # With installed 2.46.0 the same preflight passes, and without a minimum no check is made.
    monkeypatch.setitem(sys.modules, "fakesdk", types.ModuleType("fakesdk"))
    monkeypatch.setattr(credentials.importlib.metadata, "version", lambda name: "2.11.0")
    with pytest.raises(RuntimeError, match=r"fakesdk>=2\.12 \(installed: 2\.11\.0\)"):
        preflight_sdk("fakesdk", "pip install fakesdk", "2.12")
    monkeypatch.setattr(credentials.importlib.metadata, "version", lambda name: "2.46.0")
    preflight_sdk("fakesdk", "pip install fakesdk", "2.12")  # no error
    preflight_sdk("fakesdk", "pip install fakesdk")  # no floor, no check


def test_preflight_skips_floor_without_package_metadata(monkeypatch, capsys):
    # [FUNCTION DOC] Makes the metadata version lookup raise a package-not-found error. Asserts
    # preflight does not raise and prints a note on standard output saying the minimum version
    # cannot be verified.
    """A vendored copy imports fine but has no dist-info; the floor check is
    skipped with a note rather than blocking a possibly-fine install."""
    monkeypatch.setitem(sys.modules, "fakesdk", types.ModuleType("fakesdk"))

    def raise_not_found(name):
        # [FUNCTION DOC] HELPER: Replacement version lookup that always raises a package-not-found
        # error carrying the requested name.
        raise credentials.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(credentials.importlib.metadata, "version", raise_not_found)
    preflight_sdk("fakesdk", "pip install fakesdk", "2.12")  # no error
    assert "cannot verify fakesdk>=2.12" in capsys.readouterr().out


def test_version_tuple_reads_leading_numbers():
    # [FUNCTION DOC] Asserts the private version parser returns integer tuples of the leading
    # numeric components (dropping a release-candidate suffix) and that the tuples compare in
    # version order.
    assert credentials._version_tuple("2.12") == (2, 12)
    assert credentials._version_tuple("2.46.0") == (2, 46, 0)
    assert credentials._version_tuple("2.12.0rc1") == (2, 12, 0)
    assert credentials._version_tuple("2.12") < credentials._version_tuple("2.46.0")


# --- worker-side key resolution -------------------------------------------------


def test_resolve_env_value_wins_over_the_file(monkeypatch, tmp_path):
    # [FUNCTION DOC] With both the key variable and a file variable pointing at a file set,
    # asserts key resolution returns the environment value.
    key_file = tmp_path / "api_key"
    key_file.write_text("from_file\n")
    monkeypatch.setenv("PROV_API_KEY", "from_env")
    monkeypatch.setenv("PROV_API_KEY_FILE", str(key_file))
    assert resolve_provider_api_key("PROV_API_KEY", "PROV_API_KEY_FILE", "~/nope") == "from_env"


def test_resolve_falls_back_to_the_file_and_strips_whitespace(monkeypatch, tmp_path):
    # [FUNCTION DOC] With the key variable unset and the file variable pointing at a file with
    # surrounding whitespace, asserts resolution returns the stripped file content.
    key_file = tmp_path / "api_key"
    key_file.write_text("  from_file \n")
    monkeypatch.delenv("PROV_API_KEY", raising=False)
    monkeypatch.setenv("PROV_API_KEY_FILE", str(key_file))
    assert resolve_provider_api_key("PROV_API_KEY", "PROV_API_KEY_FILE", "~/nope") == "from_file"


def test_resolve_default_path_expands_the_home_dir(monkeypatch, tmp_path):
    # [FUNCTION DOC] Points the home directory variable at a temporary directory containing a key
    # file, unsets both key variables, and asserts resolution reads the default path given with a
    # home-relative prefix.
    (tmp_path / ".config").mkdir()
    (tmp_path / ".config" / "api_key").write_text("from_default\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("PROV_API_KEY", raising=False)
    monkeypatch.delenv("PROV_API_KEY_FILE", raising=False)
    assert resolve_provider_api_key("PROV_API_KEY", "PROV_API_KEY_FILE", "~/.config/api_key") == "from_default"


def test_resolve_missing_key_names_both_supplies(monkeypatch, tmp_path):
    # [FUNCTION DOC] With the key variable unset and the file variable pointing at a missing
    # file, asserts a runtime error saying the key variable is unset.
    monkeypatch.delenv("PROV_API_KEY", raising=False)
    monkeypatch.setenv("PROV_API_KEY_FILE", str(tmp_path / "absent"))
    with pytest.raises(RuntimeError, match="PROV_API_KEY is unset"):
        resolve_provider_api_key("PROV_API_KEY", "PROV_API_KEY_FILE", "~/nope")


# --- whole-provider provisioning ------------------------------------------------


def test_provision_provider_wires_key_path_addresses_and_preflight(monkeypatch, tmp_path):
    # [FUNCTION DOC] Provisions the e2b provider with a key file, one set endpoint variable, one
    # unset endpoint variable and a fake importable SDK module. Asserts the environment gets the
    # key-file path variable and the set endpoint, omits the unset endpoint, and never contains
    # the secret value.
    """The one call a launcher makes per provider: key file by path, endpoint
    vars by value, SDK importable."""
    key_file = tmp_path / "api_key"
    key_file.write_text("e2b_secret\n")
    monkeypatch.setenv("E2B_API_URL", "http://agentenv.internal:8000")
    monkeypatch.delenv("E2B_SANDBOX_URL", raising=False)
    monkeypatch.setitem(sys.modules, "e2b", types.ModuleType("e2b"))

    env: dict[str, str] = {}
    provision_provider(env, PROVIDER_CREDENTIALS["e2b"], arg_path=str(key_file))

    assert env["E2B_API_KEY_FILE"] == str(key_file)
    assert env["E2B_API_URL"] == "http://agentenv.internal:8000"
    assert "E2B_SANDBOX_URL" not in env  # unset vars are not forwarded
    assert "e2b_secret" not in str(env)


def test_modal_provisioning_uses_public_sandbox_v2_switch(monkeypatch, tmp_path):
    # [FUNCTION DOC] Provisions modal with a config file, a fake SDK module and a faked installed
    # version of 1.5.5, with the sandbox-v2 switch unset. Asserts the environment gets the config
    # path and the v2 switch defaulting to "1". Then sets the switch to "0" in the process
    # environment, provisions again, and asserts the user's value "0" is kept.
    config = tmp_path / "modal.toml"
    config.write_text('[radixark]\ntoken_id = "ak-123"\ntoken_secret = "as-456"\n')
    monkeypatch.setitem(sys.modules, "modal", types.ModuleType("modal"))
    monkeypatch.setattr(credentials.importlib.metadata, "version", lambda name: "1.5.5")
    monkeypatch.delenv("MODAL_SANDBOX_V2", raising=False)

    env: dict[str, str] = {}
    provision_provider(env, PROVIDER_CREDENTIALS["modal"], arg_path=str(config))

    assert env["MODAL_CONFIG_PATH"] == str(config)
    assert env["MODAL_SANDBOX_V2"] == "1"

    monkeypatch.setenv("MODAL_SANDBOX_V2", "0")
    overridden: dict[str, str] = {}
    provision_provider(overridden, PROVIDER_CREDENTIALS["modal"], arg_path=str(config))
    assert overridden["MODAL_SANDBOX_V2"] == "0"


def test_credential_available_accepts_either_supply(monkeypatch, tmp_path):
    # [FUNCTION DOC] Checks the availability predicate for e2b: false with no environment key and a
    # nonexistent file, false with a whitespace-only file, true with a non-empty file, and true
    # when the key is in the environment even if the file path does not exist.
    """The predicate a caller uses to decide rather than fail: a non-empty key
    file, or every one of the provider's credential variables."""
    key_file = tmp_path / "api_key"
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    assert not credential_available(PROVIDER_CREDENTIALS["e2b"], arg_path=str(key_file))

    key_file.write_text("  \n")  # a blank placeholder is not a credential
    assert not credential_available(PROVIDER_CREDENTIALS["e2b"], arg_path=str(key_file))

    key_file.write_text("e2b_secret\n")
    assert credential_available(PROVIDER_CREDENTIALS["e2b"], arg_path=str(key_file))

    monkeypatch.setenv("E2B_API_KEY", "e2b_secret")
    assert credential_available(PROVIDER_CREDENTIALS["e2b"], arg_path=str(tmp_path / "absent"))


def test_credential_available_needs_every_var_of_a_multi_var_provider(monkeypatch, tmp_path):
    # [FUNCTION DOC] For modal with an absent config file, asserts availability is false with only
    # the token id set and true once both token id and secret are set.
    """Modal's credential is a token pair: half of it is a misconfiguration,
    not a usable supply."""
    absent = str(tmp_path / "absent.toml")
    monkeypatch.setenv("MODAL_TOKEN_ID", "id")
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    assert not credential_available(PROVIDER_CREDENTIALS["modal"], arg_path=absent)

    monkeypatch.setenv("MODAL_TOKEN_SECRET", "secret")
    assert credential_available(PROVIDER_CREDENTIALS["modal"], arg_path=absent)
