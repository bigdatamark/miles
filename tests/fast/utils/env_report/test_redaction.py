# [MODULE DOC] Unit tests for the secret redaction helpers of the environment report. Covered:
# redacting secret values in a command line vector (separate and inline flag forms, multi-valued
# flags, environment-valued flags, proxy URLs with credentials), redacting a single structured
# argument, the deterministic one-way digest used as replacement (prefix "redacted-sha256:"),
# redacting environment variable mappings by name suffix, and redacting inference server info
# including per-scheduler internal states. All tests are pure functions on literal inputs.
import pytest

from miles.utils.env_report.redaction import _redact, redact_arg, redact_argv, redact_env_vars, redact_server_info


class TestRedactArgv:
    # [CLASS DOC] Tests for redacting a list of command line tokens: values of declared secret
    # flags are replaced by digests, other tokens stay unchanged and in order.
    def test_hides_the_value_of_a_secret_flag(self) -> None:
        # [FUNCTION DOC] For the weights-and-biases key flag followed by a value, the secret text
        # is absent from the result, the program name and the flag itself are kept, and the
        # following non-secret "reward key" flag and value are kept.
        """A hashed wandb_key in args is pointless while the same key sits in argv verbatim."""
        argv = redact_argv(["train.py", "--wandb-key", "s3cret", "--reward-key", "reward"])
        assert "s3cret" not in argv
        assert argv[:2] == ["train.py", "--wandb-key"]
        assert argv[-2:] == ["--reward-key", "reward"]

    def test_hides_the_value_of_an_inline_secret_flag(self) -> None:
        # [FUNCTION DOC] For the inline form (flag, equals sign, value in one token), the value
        # part is replaced and the token starts with the flag, equals sign and the digest prefix.
        argv = redact_argv(["train.py", "--wandb-key=s3cret"])
        assert "s3cret" not in argv[1]
        assert argv[1].startswith("--wandb-key=redacted-sha256:")

    def test_keeps_an_argv_without_secrets_unchanged(self) -> None:
        # [FUNCTION DOC] A vector with no secret flags (including a "reward key" flag that only
        # names a dataset column) is returned equal to the input.
        argv = ["train.py", "--reward-key", "reward", "--lr=1e-4"]
        assert redact_argv(argv) == argv

    def test_a_trailing_secret_flag_hides_nothing_that_follows(self) -> None:
        # [FUNCTION DOC] A secret flag as the last token has no value to hide, and the vector is
        # returned unchanged without error.
        assert redact_argv(["train.py", "--wandb-key"]) == ["train.py", "--wandb-key"]

    def test_hides_every_value_of_a_multi_valued_secret_flag(self) -> None:
        # [FUNCTION DOC] For the router control plane keys flag followed by two values and then
        # another flag, neither secret fragment appears in the joined result and the trailing
        # "--lr 1" tokens remain, so redaction stops at the next flag.
        """The router takes its control-plane keys as a list, so hiding only the first leaks the rest."""
        argv = redact_argv(["train.py", "--router-control-plane-api-keys", "k1:n:r:s1", "k2:n:r:s2", "--lr", "1"])

        assert "s1" not in " ".join(argv) and "s2" not in " ".join(argv)
        assert argv[-2:] == ["--lr", "1"]

    def test_hides_the_secrets_inside_an_environment_valued_flag(self) -> None:
        # [FUNCTION DOC] For the training environment variables flag whose value is a JSON
        # object text, the secret variable's value is hidden while the non-secret variable's
        # value ("INFO") remains visible in the token (index 2).
        """--train-env-vars carries a whole environment, and a credential in it is not a secret arg by name."""
        argv = redact_argv(["train.py", "--train-env-vars", '{"WANDB_API_KEY": "hunter2", "NCCL_DEBUG": "INFO"}'])

        assert "hunter2" not in argv[2]
        assert "INFO" in argv[2]

    def test_hides_the_credentials_of_a_proxy_url(self) -> None:
        # [FUNCTION DOC] For the HTTP proxy flag with a URL containing user name and password,
        # both are removed from the value, which still ends with the host and port part.
        """An authenticated proxy carries a password, and --http-proxy is not a secret flag by name."""
        argv = redact_argv(["train.py", "--http-proxy", "http://alice:hunter2@proxy:7897"])

        assert "hunter2" not in argv[2] and "alice" not in argv[2]
        assert argv[2].endswith("@proxy:7897")

    def test_keeps_a_proxy_url_without_credentials_unchanged(self) -> None:
        # [FUNCTION DOC] An inline proxy flag with a URL lacking credentials is returned
        # unchanged.
        """The host and port are the part an audit reads back, and no secret is hiding in them."""
        assert redact_argv(["train.py", "--http-proxy=http://proxy:7897"]) == [
            "train.py",
            "--http-proxy=http://proxy:7897",
        ]


class TestRedactArg:
    # [CLASS DOC] Tests for redacting one structured argument given its name and value (as used
    # when dumping the parsed argument namespace).
    def test_hides_the_credentials_the_structured_args_carry(self) -> None:
        # [FUNCTION DOC] The "http_proxy" argument with credentials in its URL has them removed
        # while the host and port suffix is kept.
        """process.args is persisted next to process.argv, so redacting one alone leaks the other."""
        redacted = redact_arg("http_proxy", "http://alice:hunter2@proxy:7897")

        assert "hunter2" not in redacted and redacted.endswith("@proxy:7897")

    @pytest.mark.parametrize("value", ['{"HF_TOKEN":', '["secret"]'])
    def test_a_malformed_or_non_object_environment_value_is_redacted_wholesale(self, value: str) -> None:
        # [FUNCTION DOC] Parametrized over a truncated JSON text and a JSON list for the
        # training environment variables flag: the token after the flag equals the digest of
        # the entire original text and the original text is not present anywhere in the result.
        # Note this goes through the command line vector function, not the single-argument one.
        """An environment value that cannot be safely inspected must be hashed as one opaque secret."""
        argv = redact_argv(["train.py", "--train-env-vars", value])

        assert argv[2] == _redact(value)
        assert value not in argv


class TestRedact:
    # [CLASS DOC] Tests for the low-level digest replacement function.
    def test_same_secret_hashes_to_same_digest(self) -> None:
        # [FUNCTION DOC] The same input gives the same output, different inputs give different
        # outputs, and the output does not contain the input.
        """Skew auditing needs to compare secrets across processes without revealing them."""
        assert _redact("hunter2") == _redact("hunter2")
        assert _redact("hunter2") != _redact("hunter3")
        assert "hunter2" not in _redact("hunter2")


class TestRedactEnvVars:
    # [CLASS DOC] Tests for redacting a mapping of environment variable names to values: names
    # ending in secret-like words (case-insensitive) have their values replaced; the result is
    # sorted by name and keeps every variable.
    @pytest.mark.parametrize(
        "name",
        [
            "WANDB_API_KEY",
            "HF_TOKEN",
            "MY_SECRET",
            "DB_PASSWORD",
            "PG_PASSWD",
            "GCP_CREDENTIALS",
            "hf_token",
            "NEON_DATABASE_URL",
        ],
    )
    def test_redacts_a_secret_named_variable(self, name: str) -> None:
        # [FUNCTION DOC] Parametrized over names ending with key, token, secret, password,
        # passwd, credentials (also lowercase) and database URL: the value "s3cret" does not
        # appear in the redacted entry.
        """Every secret-ish name suffix is redacted, whatever its case."""
        assert "s3cret" not in redact_env_vars({name: "s3cret"})[name]

    @pytest.mark.parametrize(
        "name", ["TOKENIZERS_PARALLELISM", "KEYRING_PATH", "SSH_KEY_FILE", "CUDA_VISIBLE_DEVICES"]
    )
    def test_keeps_a_variable_that_merely_contains_a_secret_word(self, name: str) -> None:
        # [FUNCTION DOC] Parametrized over names that contain a secret-like word in the middle
        # or start but do not end with one: their value "plain" is kept verbatim.
        """Hashing every name containing 'key' would erase exactly the values an audit reads."""
        assert redact_env_vars({name: "plain"})[name] == "plain"

    def test_sorts_and_keeps_every_variable(self) -> None:
        # [FUNCTION DOC] Three variables given in unsorted order come back as the same three
        # keys sorted alphabetically, and a non-secret value is unchanged.
        """The report dumps all env vars, so redaction must never drop one."""
        redacted = redact_env_vars({"ZZZ": "z", "HF_TOKEN": "t", "RANK": "3"})
        assert list(redacted.keys()) == ["HF_TOKEN", "RANK", "ZZZ"]
        assert redacted["RANK"] == "3"


class TestRedactServerInfo:
    # [CLASS DOC] Tests for redacting the information mapping returned by an inference server:
    # engine-named secret keys (no prefix) at the top level and inside every internal state, plus
    # environment mappings in each internal state. The input is never mutated, and unknown shapes
    # of the internal states entry are tolerated.
    def test_redacts_the_keys_repeated_inside_every_internal_state(self) -> None:
        # [FUNCTION DOC] Two internal states, one with an API key and a queue length and one
        # with an SSL key file password: the secrets become redacted digests while the queue
        # length stays 0.
        """Each internal state is a whole ServerArgs, so the credentials appear once per scheduler."""
        redacted = redact_server_info(
            {"internal_states": [{"api_key": "a", "waiting_queue": 0}, {"ssl_keyfile_password": "c"}]}
        )

        assert redacted["internal_states"][0]["api_key"].startswith("redacted-sha256:")
        assert redacted["internal_states"][0]["waiting_queue"] == 0
        assert redacted["internal_states"][1]["ssl_keyfile_password"].startswith("redacted-sha256:")

    def test_redacts_an_internal_state_that_carries_no_environment(self) -> None:
        # [FUNCTION DOC] An internal state with an API key but no environment variables entry
        # still has the key redacted.
        """An engine too old to answer with env_vars still repeats its ServerArgs in every state."""
        redacted = redact_server_info({"internal_states": [{"api_key": "a"}]})

        assert redacted["internal_states"][0]["api_key"].startswith("redacted-sha256:")

    def test_redacts_the_engine_keys_that_arrive_without_a_miles_prefix(self) -> None:
        # [FUNCTION DOC] Top-level API key, admin API key and SSL key file password are
        # redacted; the model path stays "/models/qwen".
        """The engine reports its own ServerArgs, whose names lack the sglang_ prefix miles args carry."""
        redacted = redact_server_info(
            {"api_key": "a", "admin_api_key": "b", "ssl_keyfile_password": "c", "model_path": "/models/qwen"}
        )

        assert all(redacted[name].startswith("redacted-sha256:") for name in ("api_key", "admin_api_key"))
        assert redacted["ssl_keyfile_password"].startswith("redacted-sha256:")
        assert redacted["model_path"] == "/models/qwen"

    def test_redacts_the_environment_of_every_scheduler(self) -> None:
        # [FUNCTION DOC] Two internal states with environment mappings: the token and API key
        # variables are redacted in their respective states while the rank variable stays "0".
        """internal_states holds one entry per DP scheduler, and each carries a whole environment."""
        redacted = redact_server_info(
            {
                "internal_states": [
                    {"env_vars": {"HF_TOKEN": "t0", "RANK": "0"}},
                    {"env_vars": {"WANDB_API_KEY": "k1"}},
                ]
            }
        )

        assert redacted["internal_states"][0]["env_vars"]["HF_TOKEN"].startswith("redacted-sha256:")
        assert redacted["internal_states"][0]["env_vars"]["RANK"] == "0"
        assert redacted["internal_states"][1]["env_vars"]["WANDB_API_KEY"].startswith("redacted-sha256:")

    def test_keeps_an_internal_state_that_carries_no_environment(self) -> None:
        # [FUNCTION DOC] Server info with nothing secret (a state with only a queue length and a
        # version string) is returned equal to the input.
        """An engine built before the env var gate existed reports no env_vars key at all."""
        server_info = {"internal_states": [{"waiting_queue": 0}], "version": "0.5.0"}
        assert redact_server_info(server_info) == server_info

    def test_leaves_the_caller_copy_untouched(self) -> None:
        # [FUNCTION DOC] After redacting, the original input still holds the plain API key and
        # token values (a copy is redacted, not the input).
        """The raw response is the engine's answer, and redaction must not rewrite it under the caller."""
        server_info = {"api_key": "a", "internal_states": [{"env_vars": {"HF_TOKEN": "t"}}]}

        redact_server_info(server_info)

        assert server_info["api_key"] == "a"
        assert server_info["internal_states"][0]["env_vars"]["HF_TOKEN"] == "t"

    def test_ignores_an_internal_states_shape_it_does_not_know(self) -> None:
        # [FUNCTION DOC] When the internal states entry is none, or a list containing a plain
        # string, the input is returned equal without raising.
        """The engine's schema is not miles's to control, and an unexpected shape must not raise."""
        assert redact_server_info({"internal_states": None}) == {"internal_states": None}
        assert redact_server_info({"internal_states": ["opaque"]}) == {"internal_states": ["opaque"]}
