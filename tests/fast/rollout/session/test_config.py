# [MODULE DOC] Tests of the session server configuration model and of the function computing it
# from launch arguments. The module-level tables are: a mapping from each launch argument name to
# the config field it feeds, the fields supplied by the caller (host, port, instance id, backend
# URL), the optional launch arguments that may be absent (read as none), a set of distinct argument
# values no default would produce, and a complete keyword set for constructing the config directly.
# Covered: every config field has a known source, call-site fields copied, each argument field
# copied one to one, the boolean/string session server flag, absent optional arguments, a default
# rollout temperature for older arguments, and the model's strictness (all fields required, unknown
# fields rejected, frozen after construction).
from __future__ import annotations

from argparse import Namespace

import pytest
from pydantic import ValidationError

from miles.rollout.session.config import SessionServerConfig, compute_session_server_config

_ARGS_TO_CONFIG_FIELD = {
    "miles_router_timeout": "timeout",
    "hf_checkpoint": "hf_checkpoint",
    "chat_template_path": "chat_template_path",
    "tito_model": "tito_model",
    "apply_chat_template_kwargs": "apply_chat_template_kwargs",
    "use_rollout_routing_replay": "use_rollout_routing_replay",
    "use_rollout_indexer_replay": "use_rollout_indexer_replay",
    "use_sampling_support_replay": "use_sampling_support_replay",
    "sglang_speculative_algorithm": "sglang_speculative_algorithm",
    "num_layers": "num_layers",
    "moe_router_topk": "moe_router_topk",
    "save_debug_trajectory_data": "save_debug_trajectory_data",
    "lora_rank": "lora_rank",
    "lora_adapter_path": "lora_adapter_path",
    "lora_train_only": "lora_train_only",
    "use_session_server": "use_session_server",
    "session_message_matcher": "session_message_matcher",
    "pause_generation_mode": "pause_generation_mode",
    "session_sample_picker_path": "session_sample_picker_path",
    "session_sample_postprocessor_path": "session_sample_postprocessor_path",
    "rollout_top_logprobs_num": "rollout_top_logprobs_num",
    "rollout_sampling_logprobs_mode": "rollout_sampling_logprobs_mode",
    "rollout_temperature": "rollout_temperature",
}

_CALL_SITE_FIELDS = ("host", "port", "instance_id", "backend_url")
_OPTIONAL_ARGS_ATTRS = (
    "num_layers",
    "pause_generation_mode",
    "moe_router_topk",
    "use_session_server",
    "session_sample_picker_path",
    "session_sample_postprocessor_path",
)

_DISTINCT_ARGS_VALUES = dict(
    miles_router_timeout=31.5,
    hf_checkpoint="/fake/model",
    chat_template_path="/fake/chat_template.jinja",
    tito_model="tito-xyz",
    apply_chat_template_kwargs={"enable_thinking": True},
    use_rollout_routing_replay=True,
    use_rollout_indexer_replay=True,
    use_sampling_support_replay=True,
    sglang_speculative_algorithm="EAGLE",
    num_layers=61,
    moe_router_topk=8,
    save_debug_trajectory_data="/fake/traj",
    lora_rank=32,
    lora_adapter_path="/fake/adapters/x",
    lora_train_only=True,
    use_session_server="v2",
    session_message_matcher="fake.matcher",
    pause_generation_mode="in_place",
    session_sample_picker_path="fake.picker",
    session_sample_postprocessor_path="fake.postprocessor",
    rollout_top_logprobs_num=32,
    rollout_sampling_logprobs_mode="support",
    rollout_temperature=0.7,
)


def _make_args(**overrides) -> Namespace:
    # [FUNCTION DOC] HELPER: Builds a launch arguments namespace from the distinct values table,
    # with the given keyword overrides replacing or adding entries.
    defaults = dict(_DISTINCT_ARGS_VALUES)
    defaults.update(overrides)
    return Namespace(**defaults)


class TestComputeSessionServerConfig:
    # [CLASS DOC] Tests of computing the config from launch arguments and call-site values.
    def test_every_config_field_has_a_known_source(self):
        # [FUNCTION DOC] Asserts the union of call-site fields and the mapped config fields equals
        # exactly the model's field set, so adding a config field without extending the mapping
        # fails.
        """Adding a config field without extending this test's mapping must fail here."""
        covered = set(_CALL_SITE_FIELDS) | set(_ARGS_TO_CONFIG_FIELD.values())
        assert covered == set(SessionServerConfig.model_fields)

    def test_call_site_fields_are_copied(self):
        # [FUNCTION DOC] Computes the config with explicit host, port, instance id and backend
        # URL and asserts the config carries exactly those values.
        """Host, port, instance id, and backend url come from the caller, not from args."""
        config = compute_session_server_config(
            _make_args(), host="10.0.0.1", port=5001, instance_id="abc", backend_url="http://10.0.0.2:3000"
        )
        assert config.host == "10.0.0.1"
        assert config.port == 5001
        assert config.instance_id == "abc"
        assert config.backend_url == "http://10.0.0.2:3000"

    @pytest.mark.parametrize("args_attr,config_field", sorted(_ARGS_TO_CONFIG_FIELD.items()))
    def test_each_args_field_is_copied(self, args_attr: str, config_field: str):
        # [FUNCTION DOC] Parametrized over every (argument name, config field) pair in the
        # mapping (sorted). Computes the config from the distinct-values arguments and asserts the
        # config field equals the distinct value of the argument. Note the router timeout argument
        # maps to the timeout field.
        """Every args-derived field maps one-to-one, using values no hardcoded default would produce."""
        config = compute_session_server_config(
            _make_args(), host="10.0.0.1", port=5001, instance_id="abc", backend_url="http://10.0.0.2:3000"
        )
        assert getattr(config, config_field) == _DISTINCT_ARGS_VALUES[args_attr]

    @pytest.mark.parametrize("flag_value", [True, False])
    def test_a_boolean_session_server_flag_reaches_the_config_unchanged(self, flag_value: bool):
        # [FUNCTION DOC] Parametrized over true and false for the session server argument.
        # Computes the config and asserts the config field is that same boolean (identity).
        """A bare --use-session-server (True) and an omitted one (False) both survive the launch-time config build."""
        config = compute_session_server_config(
            _make_args(use_session_server=flag_value),
            host="10.0.0.1",
            port=5001,
            instance_id="abc",
            backend_url="http://10.0.0.2:3000",
        )
        assert config.use_session_server is flag_value

    def test_missing_optional_args_fall_back_to_none(self):
        # [FUNCTION DOC] Builds an arguments namespace without the optional attributes (number of
        # layers, pause mode, router top-k, session server, picker path, postprocessor path),
        # computes the config, and asserts those config fields are all none.
        """An args object that omits the optional attributes yields None for them instead of failing."""
        present = {name: value for name, value in _DISTINCT_ARGS_VALUES.items() if name not in _OPTIONAL_ARGS_ATTRS}
        config = compute_session_server_config(
            Namespace(**present), host="10.0.0.1", port=5001, instance_id="abc", backend_url="http://10.0.0.2:3000"
        )
        assert [getattr(config, name) for name in _OPTIONAL_ARGS_ATTRS] == [None] * len(_OPTIONAL_ARGS_ATTRS)

    def test_older_args_default_rollout_temperature(self):
        # [FUNCTION DOC] Builds arguments lacking the rollout temperature (as older launches
        # would) and asserts the config's rollout temperature defaults to 1.0.
        args = Namespace(
            **{key: value for key, value in _DISTINCT_ARGS_VALUES.items() if key != "rollout_temperature"}
        )
        config = compute_session_server_config(
            args, host="127.0.0.1", port=5001, instance_id=None, backend_url="http://127.0.0.1:3000"
        )
        assert config.rollout_temperature == 1.0


_COMPLETE_CONFIG_KWARGS = dict(
    host="127.0.0.1",
    port=5001,
    instance_id=None,
    backend_url="http://127.0.0.1:3000",
    timeout=None,
    hf_checkpoint=None,
    chat_template_path=None,
    tito_model="default",
    apply_chat_template_kwargs=None,
    use_rollout_routing_replay=False,
    use_rollout_indexer_replay=False,
    use_sampling_support_replay=False,
    sglang_speculative_algorithm=None,
    num_layers=None,
    moe_router_topk=None,
    save_debug_trajectory_data=None,
    lora_rank=0,
    lora_adapter_path=None,
    lora_train_only=False,
    use_session_server=None,
    session_message_matcher="strict",
    pause_generation_mode=None,
    session_sample_picker_path=None,
    session_sample_postprocessor_path=None,
    rollout_top_logprobs_num=0,
    rollout_sampling_logprobs_mode="selected",
    rollout_temperature=1.0,
)


class TestSessionServerConfig:
    # [CLASS DOC] Tests of the config model's own validation rules.
    def test_complete_kwargs_cover_every_field(self):
        # [FUNCTION DOC] Asserts the complete keyword table has exactly the model's fields and
        # that constructing the config from it succeeds (port 5001).
        """The per-field omission cases are only meaningful if the full kwargs validate."""
        assert set(_COMPLETE_CONFIG_KWARGS) == set(SessionServerConfig.model_fields)
        assert SessionServerConfig(**_COMPLETE_CONFIG_KWARGS).port == 5001

    def test_the_bare_session_server_flag_is_accepted(self):
        # [FUNCTION DOC] A session server flag of true (a bare command-line flag) is accepted and
        # kept as true.
        """--use-session-server written without a version arrives as True, which used to fail validation."""
        config = SessionServerConfig(**{**_COMPLETE_CONFIG_KWARGS, "use_session_server": True})

        assert config.use_session_server is True

    def test_the_omitted_session_server_flag_is_accepted(self):
        # [FUNCTION DOC] A session server flag of false (flag omitted) is accepted and kept as
        # false.
        """Leaving --use-session-server off yields False, which the str-only annotation used to reject."""
        config = SessionServerConfig(**{**_COMPLETE_CONFIG_KWARGS, "use_session_server": False})

        assert config.use_session_server is False

    @pytest.mark.parametrize("version", ["v1", "v2"])
    def test_a_named_session_server_version_stays_a_string(self, version: str):
        # [FUNCTION DOC] Parametrized over "v1" and "v2". Asserts the stored flag equals the
        # string (it is not coerced to a boolean).
        """Widening the flag to accept booleans must not turn a named version into a bool."""
        config = SessionServerConfig(**{**_COMPLETE_CONFIG_KWARGS, "use_session_server": version})

        assert config.use_session_server == version

    @pytest.mark.parametrize("value", [3.5, ["v2"], {"version": "v2"}])
    def test_a_session_server_flag_of_another_type_is_rejected(self, value: object):
        # [FUNCTION DOC] Parametrized over a float, a list and a mapping. Asserts constructing the
        # config with such a flag raises a validation error.
        """The flag accepts only booleans and version strings, so other types must still fail validation."""
        with pytest.raises(ValidationError):
            SessionServerConfig(**{**_COMPLETE_CONFIG_KWARGS, "use_session_server": value})

    @pytest.mark.parametrize("missing", sorted(SessionServerConfig.model_fields))
    def test_every_field_is_required(self, missing: str):
        # [FUNCTION DOC] Parametrized over every model field. Constructing the config from the
        # complete keyword table without that one field raises a validation error.
        """Omitting any single field must fail, so none can be silently defaulted."""
        kwargs = {name: value for name, value in _COMPLETE_CONFIG_KWARGS.items() if name != missing}
        with pytest.raises(ValidationError):
            SessionServerConfig(**kwargs)

    def test_unknown_field_is_rejected(self):
        # [FUNCTION DOC] Constructing the config with an extra unknown keyword raises a validation
        # error.
        """An unrecognized keyword must fail validation instead of being silently ignored."""
        with pytest.raises(ValidationError):
            SessionServerConfig(**_COMPLETE_CONFIG_KWARGS, unknown_field="whatever")

    def test_field_assignment_is_rejected(self):
        # [FUNCTION DOC] Assigning a new port to a constructed config raises a validation error
        # (the model is frozen).
        """The config is frozen, so assigning to a field after construction must fail."""
        config = SessionServerConfig(**_COMPLETE_CONFIG_KWARGS)
        with pytest.raises(ValidationError):
            config.port = 5002
