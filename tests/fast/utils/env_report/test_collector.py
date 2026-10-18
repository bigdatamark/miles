# [MODULE DOC] Unit tests for the environment report collector. Covered: parsing entries of the
# pip inspect output, detecting editable installs, probing pip (success, failure, exception, and
# which environment it is run with), taking a snapshot of the process environment for the probe,
# dumping the command line arguments (serialization, skipping, secret redaction, deep copying),
# collecting key package versions, and assembling the full structured report including process
# identity, redacted environment variables and the launcher record. The pip subprocess is mocked
# via fixtures from the shared conftest; environment variables are patched temporarily.
import json
import os
import subprocess
import sys
import types
from unittest.mock import patch


from tests.fast.utils.env_report.conftest import SAMPLE_PIP_INSPECT, make_args

from miles.utils.audit_utils.event_logger.models import EnvReport, EnvReportEditablePackageInfo
from miles.utils.env_report.collector import (
    _collect_key_versions,
    _collect_pip_info,
    _dump_args,
    _is_editable,
    _parse_pip_entry,
    collect_env_report,
    collect_env_report_snapshot,
)
from miles.utils.env_report.launcher_report import LAUNCHER_REPORT_ENV_VAR


class TestParsePipEntry:
    # [CLASS DOC] Tests for reducing one installed-package entry of the pip inspect output to a
    # dictionary with name and version strings.
    def test_normal_package(self) -> None:
        # [FUNCTION DOC] The metadata name and version are copied into the result.
        entry = _parse_pip_entry({"metadata": {"name": "torch", "version": "2.5.0"}})
        assert entry == {"name": "torch", "version": "2.5.0"}

    def test_missing_metadata(self) -> None:
        # [FUNCTION DOC] An entry without metadata gives empty strings for both fields rather
        # than failing.
        entry = _parse_pip_entry({})
        assert entry == {"name": "", "version": ""}


class TestIsEditable:
    # [CLASS DOC] Tests for the check whether an installed-package entry is an editable install,
    # which is true only when its direct URL record has a directory info section flagged editable.
    def test_editable_package(self) -> None:
        # [FUNCTION DOC] A direct URL with directory info marked editable is editable.
        pkg = {"direct_url": {"url": "file:///workspace/miles", "dir_info": {"editable": True}}}
        assert _is_editable(pkg) is True

    def test_non_editable_package(self) -> None:
        # [FUNCTION DOC] An entry without any direct URL is not editable.
        assert _is_editable({"metadata": {"name": "torch"}}) is False

    def test_archive_url_not_editable(self) -> None:
        # [FUNCTION DOC] A direct URL pointing at an archive (archive info instead of directory
        # info) is not editable.
        pkg = {"direct_url": {"url": "https://example.com/foo.tar.gz", "archive_info": {}}}
        assert _is_editable(pkg) is False


class TestCollectPipInfo:
    # [CLASS DOC] Tests for probing pip: the collector runs the pip inspect subprocess (mocked
    # here) with a given environment mapping and returns a record with the editable packages
    # (name, version, location), the full list of name/version pairs, and a flag telling whether
    # the probe succeeded.
    def test_parses_pip_inspect_output(self) -> None:
        # [FUNCTION DOC] With the sample output the probe flag is true, the full list has four
        # entries in order (first miles, third torch), and exactly two editable packages (miles
        # and sglang) are reported with the file URL's path as location.
        mock_result = subprocess.CompletedProcess(
            args=["pip", "inspect"],
            returncode=0,
            stdout=json.dumps(SAMPLE_PIP_INSPECT),
            stderr="",
        )
        with patch("miles.utils.env_report.collector.subprocess.run", return_value=mock_result):
            pip_info = _collect_pip_info({})

        editable, full_list = pip_info.editable_packages, pip_info.full_pip_list
        assert pip_info.probed is True
        assert len(full_list) == 4
        assert full_list[0] == {"name": "miles", "version": "0.2.1"}
        assert full_list[2] == {"name": "torch", "version": "2.5.0"}

        assert len(editable) == 2
        assert editable[0] == EnvReportEditablePackageInfo(
            name="miles",
            version="0.2.1",
            location="/workspace/miles",
        )
        assert editable[1] == EnvReportEditablePackageInfo(
            name="sglang",
            version="0.4.0",
            location="/workspace/sglang",
        )

    def test_pip_inspect_failure_returns_empty(self) -> None:
        # [FUNCTION DOC] A non-zero exit code yields empty editable and full lists and a probe
        # flag of false.
        mock_result = subprocess.CompletedProcess(
            args=["pip", "inspect"],
            returncode=1,
            stdout="",
            stderr="error",
        )
        with patch("miles.utils.env_report.collector.subprocess.run", return_value=mock_result):
            pip_info = _collect_pip_info({})
        assert pip_info.editable_packages == []
        assert pip_info.full_pip_list == []
        assert pip_info.probed is False

    def test_pip_inspect_exception_returns_empty(self) -> None:
        # [FUNCTION DOC] If running the subprocess raises an operating system error, the
        # exception is swallowed and the result is the same empty, not-probed record.
        with patch("miles.utils.env_report.collector.subprocess.run", side_effect=OSError("no pip")):
            pip_info = _collect_pip_info({})
        assert pip_info.editable_packages == []
        assert pip_info.full_pip_list == []
        assert pip_info.probed is False

    def test_runs_pip_with_the_environment_it_is_given(self) -> None:
        # [FUNCTION DOC] The mapping passed in is handed unchanged to the subprocess call as its
        # environment keyword argument (the process environment is not read at that point).
        """The caller hands in a snapshot; reading os.environ here would race whoever mutates it."""
        mock_result = subprocess.CompletedProcess(
            args=["pip", "inspect"],
            returncode=0,
            stdout=json.dumps(SAMPLE_PIP_INSPECT),
            stderr="",
        )
        with patch("miles.utils.env_report.collector.subprocess.run", return_value=mock_result) as mock_run:
            _collect_pip_info({"HOME": "/root"})

        assert mock_run.call_args.kwargs["env"] == {"HOME": "/root"}


class TestCollectProcessEnvSnapshot:
    # [CLASS DOC] Tests for the snapshot step that captures, on the calling thread, the
    # environment mapping later used for the pip probe.
    def test_excludes_pythonpath_from_the_pip_probe(self) -> None:
        # [FUNCTION DOC] With the Python path variable set in the process, the snapshot's probe
        # environment does not contain it.
        """pip inspect misses editable packages whose source is on the PYTHONPATH."""
        with patch.dict(os.environ, {"PYTHONPATH": "/workspace/Megatron-LM"}):
            snapshot = collect_env_report_snapshot(make_args())

        assert "PYTHONPATH" not in snapshot.probe_env

    def test_snapshots_the_environment_the_probe_will_run_with(self) -> None:
        # [FUNCTION DOC] A marker variable set to "before" is captured; changing it to "after"
        # once the snapshot exists does not alter the snapshot (a copy was taken).
        """The probe runs on another thread, while the caller is still setting RANK and friends."""
        with patch.dict(os.environ, {"MILES_TEST_ENV_REPORT_MARK": "before"}):
            snapshot = collect_env_report_snapshot(make_args())
            os.environ["MILES_TEST_ENV_REPORT_MARK"] = "after"

            assert snapshot.probe_env["MILES_TEST_ENV_REPORT_MARK"] == "before"


class TestDumpArgs:
    # [CLASS DOC] Tests for turning an argument namespace into a record with a values mapping and
    # the list of names that were skipped because they cannot be represented as JSON. Declared
    # secret arguments are replaced by a hash string with the prefix "redacted-sha256:".
    def test_keeps_serializable_values(self) -> None:
        # [FUNCTION DOC] Floats, lists, nested mappings, booleans and none are kept as they are
        # and nothing is skipped.
        dump = _dump_args(make_args(lr=1e-4, tags=["a"], nested={"x": 1}, flag=True, missing=None))
        assert dump.skipped_names == []
        assert dump.values["lr"] == 1e-4
        assert dump.values["tags"] == ["a"]
        assert dump.values["nested"] == {"x": 1}
        assert dump.values["missing"] is None

    def test_skips_unserializable_values(self) -> None:
        # [FUNCTION DOC] An argument holding a plain object is left out of the values and its
        # name is listed as skipped; other arguments are unaffected.
        """A non-JSON arg is skipped by name instead of being coerced to a lossy string."""
        dump = _dump_args(make_args(model=object(), lr=1.0))
        assert dump.skipped_names == ["model"]
        assert "model" not in dump.values
        assert dump.values["lr"] == 1.0

    def test_redacts_a_declared_secret_arg(self) -> None:
        # [FUNCTION DOC] The weights-and-biases key argument is replaced by a redacted hash
        # string.
        dump = _dump_args(make_args(wandb_key="abc"))
        assert dump.values["wandb_key"].startswith("redacted-sha256:")

    def test_keeps_an_unset_secret_arg_as_none(self) -> None:
        # [FUNCTION DOC] A declared secret argument whose value is none stays none (no hashing).
        """--wandb-key defaults to None, and hashing that would crash every process at startup."""
        assert _dump_args(make_args(wandb_key=None)).values["wandb_key"] is None

    def test_redacts_every_entry_of_a_list_valued_secret_arg(self) -> None:
        # [FUNCTION DOC] For a secret argument holding a list of strings, every entry of the
        # list is replaced by a redacted hash.
        """The router's control-plane keys arrive as a list, which a string-only redaction would pass through."""
        dump = _dump_args(make_args(router_control_plane_api_keys=["k1:n:r:s1", "k2:n:r:s2"]))

        assert all(entry.startswith("redacted-sha256:") for entry in dump.values["router_control_plane_api_keys"])

    def test_redacts_the_secrets_inside_an_environment_valued_arg(self) -> None:
        # [FUNCTION DOC] For the training environment variables argument (a mapping of variable
        # name to value), the API key variable is redacted while an unrelated debug variable
        # keeps its value; redaction here depends on the variable name.
        """--train-env-vars is a whole environment, so it is redacted by variable name, not by arg name."""
        dump = _dump_args(make_args(train_env_vars={"WANDB_API_KEY": "hunter2", "NCCL_DEBUG": "INFO"}))

        assert dump.values["train_env_vars"]["WANDB_API_KEY"].startswith("redacted-sha256:")
        assert dump.values["train_env_vars"]["NCCL_DEBUG"] == "INFO"

    def test_snapshots_nested_values_instead_of_referencing_them(self) -> None:
        # [FUNCTION DOC] Appending to the original list after the dump was made does not change
        # the dumped list (values are copied).
        """The dump outlives the caller, which is free to keep mutating its own args."""
        tags = ["a"]
        dump = _dump_args(make_args(tags=tags))
        tags.append("b")
        assert dump.values["tags"] == ["a"]

    def test_keeps_dataset_column_args_that_merely_end_in_key(self) -> None:
        # [FUNCTION DOC] Arguments whose names merely end in "key" (the reward key and input key
        # dataset column names) are not treated as secrets and keep their values.
        """--reward-key names a dataset column; hashing it would hide the run's actual configuration."""
        dump = _dump_args(make_args(reward_key="reward", input_key="prompt"))
        assert dump.values["reward_key"] == "reward"
        assert dump.values["input_key"] == "prompt"

    def test_dump_is_json_serializable(self) -> None:
        # [FUNCTION DOC] Converting the dump to a plain dictionary and encoding then decoding it
        # as JSON works even when an unserializable argument was present, and keeps the float.
        dump = _dump_args(make_args(model=object(), lr=1.0))
        assert json.loads(json.dumps(dump.model_dump()))["values"]["lr"] == 1.0


class TestCollectKeyVersions:
    # [CLASS DOC] Tests for collecting a mapping of key component versions: the Python version
    # (major.minor.micro), the platform, known packages looked up in the given package list
    # case-insensitively, and torch and its CUDA version read from already-imported modules.
    def test_reports_python_and_known_packages(self) -> None:
        # [FUNCTION DOC] Python version equals the first three parts of the interpreter version,
        # the package "SGLang" (mixed case) is reported under lowercase "sglang", and a platform
        # entry exists.
        versions = _collect_key_versions(
            [{"name": "torch", "version": "2.5.0"}, {"name": "SGLang", "version": "0.4.0"}]
        )
        assert versions["python"] == ".".join(str(part) for part in sys.version_info[:3])
        assert versions["sglang"] == "0.4.0"
        assert "platform" in versions

    def test_ignores_unknown_packages(self) -> None:
        # [FUNCTION DOC] A package not on the known list (numpy) is not reported.
        versions = _collect_key_versions([{"name": "numpy", "version": "1.26.0"}])
        assert "numpy" not in versions

    def test_reports_torch_cuda_when_torch_is_imported(self) -> None:
        # [FUNCTION DOC] With a fake torch module (version 2.5.0, CUDA 12.4) placed in the
        # module table and an empty package list, the torch version and torch CUDA version are
        # reported from the module itself.
        torch = types.SimpleNamespace(__version__="2.5.0", version=types.SimpleNamespace(cuda="12.4"))
        with patch.dict(sys.modules, {"torch": torch}):
            versions = _collect_key_versions([])

        assert versions["torch"] == "2.5.0"
        assert versions["torch_cuda"] == "12.4"

    def test_reports_an_empty_cuda_version_for_a_cpu_torch(self) -> None:
        # [FUNCTION DOC] A fake torch whose CUDA version is none is reported with an empty
        # string for the torch CUDA entry.
        torch = types.SimpleNamespace(__version__="2.5.0", version=types.SimpleNamespace(cuda=None))
        with patch.dict(sys.modules, {"torch": torch}):
            assert _collect_key_versions([])["torch_cuda"] == ""

    def test_reports_nothing_about_an_unimported_torch(self) -> None:
        # [FUNCTION DOC] With the torch entry in the module table set to none (as for a module
        # that is not imported), no torch CUDA entry is reported.
        """torch is read from sys.modules, so an unimported torch costs nothing and reports nothing."""
        with patch.dict(sys.modules, {"torch": None}):
            assert "torch_cuda" not in _collect_key_versions([])


class TestCollectEnvReport:
    # [CLASS DOC] End-to-end tests of assembling the full report from a snapshot: structure,
    # package probe flag, process identity, redacted environment, launcher record handling, key
    # versions and JSON serialization. Most tests use the mocked pip probe fixture.
    def _collect(self, **overrides) -> EnvReport:
        # [FUNCTION DOC] HELPER: Takes a snapshot with an argument namespace built from the
        # overrides (see the shared factory) and returns the report collected from it.
        return collect_env_report(snapshot=collect_env_report_snapshot(make_args(**overrides)))

    def _with_record(self, tmp_path, record: str = '{"flavor": "test"}', **overrides) -> EnvReport:
        # [FUNCTION DOC] HELPER: Writes the given record text (default a JSON object with flavor
        # "test") to a file named launch-1.json in the temporary directory, then collects a
        # report whose environment report argument points at that file. Extra overrides are
        # forwarded as arguments. Returns the report.
        path = tmp_path / "launch-1.json"
        path.write_text(record)
        return self._collect(env_report=str(path), **overrides)

    def test_returns_structured_report(self, mocked_pip_inspect, tmp_path) -> None:
        # [FUNCTION DOC] The result is an environment report whose launcher record equals the
        # file content, with two editable packages and four packages in the full list.
        report = self._with_record(tmp_path)

        assert isinstance(report, EnvReport)
        assert report.process.launcher_env_report == {"flavor": "test"}
        assert len(report.editable_packages) == 2
        assert len(report.full_pip_list) == 4

    def test_says_it_probed_the_packages_it_listed(self, mocked_pip_inspect) -> None:
        # [FUNCTION DOC] After a successful mocked probe the report's packages-probed flag is
        # true.
        """An empty package list means nothing to an audit that cannot tell a failed probe from it."""
        assert self._collect().packages_probed is True

    def test_says_it_did_not_probe_when_the_probe_failed(self, mocked_pip_inspect) -> None:
        # [FUNCTION DOC] When the subprocess raises an operating system error (overriding the
        # fixture's patch), the full package list is empty and the packages-probed flag is false.
        """A failed pip inspect returns no package, exactly as an empty environment does."""
        with patch("miles.utils.env_report.collector.subprocess.run", side_effect=OSError("no pip")):
            report = self._collect()

        assert report.full_pip_list == []
        assert report.packages_probed is False

    def test_records_process_identity_context(self, mocked_pip_inspect) -> None:
        # [FUNCTION DOC] The report has a non-empty host name, the process command line equal
        # to the current interpreter's argument vector, and the dumped arguments including the
        # learning rate given.
        """The audit needs to know which host and command line produced this report."""
        report = self._collect(lr=1.0)
        assert report.process.hostname
        assert report.process.argv == sys.argv
        assert report.process.args.values["lr"] == 1.0

    def test_records_redacted_environment(self, mocked_pip_inspect) -> None:
        # [FUNCTION DOC] With a token-like and a rank-like variable set in the process
        # environment, the report keeps the rank value "7" but the token entry does not contain
        # the secret text.
        with patch.dict(os.environ, {"MILES_TEST_ENV_REPORT_TOKEN": "hunter2", "MILES_TEST_ENV_REPORT_RANK": "7"}):
            report = self._collect()

        assert report.process.env_vars["MILES_TEST_ENV_REPORT_RANK"] == "7"
        assert "hunter2" not in report.process.env_vars["MILES_TEST_ENV_REPORT_TOKEN"]

    def test_leaves_the_launcher_record_out_of_the_environment_dump(self, mocked_pip_inspect, tmp_path) -> None:
        # [FUNCTION DOC] With the launcher report environment variable set to the record path,
        # the variable is absent from the report's environment dump while the record content is
        # still stored in full.
        """The record is already stored in full, so naming its path twice only adds noise."""
        path = tmp_path / "launch-1.json"
        with patch.dict(os.environ, {LAUNCHER_REPORT_ENV_VAR: str(path)}):
            report = self._with_record(tmp_path)

        assert LAUNCHER_REPORT_ENV_VAR not in report.process.env_vars
        assert report.process.launcher_env_report == {"flavor": "test"}

    def test_records_key_versions(self, mocked_pip_inspect) -> None:
        # [FUNCTION DOC] The report's key versions include the sglang version from the sample
        # package list (0.4.0).
        report = self._collect()
        assert report.key_versions["sglang"] == "0.4.0"

    def test_a_run_with_no_launcher_records_no_launcher_report(self, mocked_pip_inspect) -> None:
        # [FUNCTION DOC] An empty environment report argument yields a launcher record of none.
        """A run started by hand names no record, and that is not a failure."""
        assert self._collect(env_report="").process.launcher_env_report is None

    def test_a_record_that_cannot_be_read_leaves_the_rest_of_the_report_intact(
        self, mocked_pip_inspect, tmp_path
    ) -> None:
        # [FUNCTION DOC] Pointing the argument at a nonexistent file gives a launcher record of
        # none without raising, and the other parts (such as the host name) are still filled in.
        """Everything else in the report is still worth having when the shared disk lost the record."""
        report = self._collect(env_report=str(tmp_path / "gone.json"))

        assert report.process.launcher_env_report is None
        assert report.process.hostname

    def test_report_serializable(self, mocked_pip_inspect, tmp_path) -> None:
        # [FUNCTION DOC] With a record file and an unserializable argument, encoding the report
        # as JSON and decoding it shows the first editable package is "miles" and the skipped
        # argument names are exactly the model argument.
        report = self._with_record(tmp_path, record='{"x": 1}', model=object())
        parsed = json.loads(report.model_dump_json())
        assert parsed["editable_packages"][0]["name"] == "miles"
        assert parsed["process"]["args"]["skipped_names"] == ["model"]
