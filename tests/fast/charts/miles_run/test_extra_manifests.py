# [MODULE DOC]
# Tests (helm rendering of the run chart) of the extra manifests value: raw YAML text strings that
# the chart installs next to the run without interpretation. Covers default none, single and
# multi-document strings, equivalence of separate and joined entries, byte-for-byte text pass-through
# (comments, key order, unevaluated template braces), deterministic output, rejection of non-string
# entries, empty and blank documents ignored, and that objects keep distinct identity by namespace
# and by API group (checked via the launcher's manifest parser). Sample manifests are module
# constants: a headless service, a stateful set, a commented config map, a templated config map, a
# notes config map in the run namespace and in another namespace, and a same-named service in a
# custom API group.
import yaml
from tests.fast.charts.utils import (
    NAMESPACE,
    extra_manifests_args,
    objects_added_by,
    render_run,
    render_run_error,
    requires_helm,
    run_helm_template_run,
)

from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import Manifest

SERVICE = """apiVersion: v1
kind: Service
metadata:
  name: external-sglang
spec:
  clusterIP: None
  selector:
    app: external-sglang
  ports:
    - name: http
      port: 30000
"""

STATEFUL_SET = """apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: external-sglang
spec:
  replicas: 2
  serviceName: external-sglang
  selector:
    matchLabels:
      app: external-sglang
  template:
    metadata:
      labels:
        app: external-sglang
    spec:
      containers:
        - name: sglang
          image: lmsysorg/sglang:latest
"""

COMMENTED = """# the engines this run talks to are not launched by miles
apiVersion: v1
kind: ConfigMap
metadata:
  name: external-sglang-notes
data:
  zzz: "written first on purpose"
  aaa: "written second on purpose"
"""

TEMPLATED = """apiVersion: v1
kind: ConfigMap
metadata:
  name: external-sglang-release
data:
  release: "{{ .Release.Name }}"
"""

NOTES = """apiVersion: v1
kind: ConfigMap
metadata:
  name: external-sglang-notes
data:
  where: "the namespace this run installs into"
"""

ELSEWHERE = """apiVersion: v1
kind: ConfigMap
metadata:
  name: external-sglang-notes
  namespace: other
data:
  where: "somewhere else entirely"
"""

OTHER_GROUP_SERVICE = """apiVersion: example.com/v1
kind: Service
metadata:
  name: external-sglang
spec:
  clusterIP: None
"""


@requires_helm
class TestExtraManifestsArePassedThrough:
    # [CLASS DOC] Pass-through behavior tests (need helm).
    def test_a_run_that_names_none_installs_nothing_beside_itself(self):
        # [FUNCTION DOC] With no extra manifests, the objects added over the baseline are empty.
        """The default has to stay an empty release, or every run would gain an object it never asked for."""
        assert objects_added_by() == []

    def test_a_single_document_is_installed_beside_the_run(self):
        # [FUNCTION DOC] One service manifest adds exactly that one parsed object.
        """This is the whole point: an object the run needs but does not describe, installed with it."""
        assert objects_added_by(SERVICE) == [yaml.safe_load(SERVICE)]

    def test_every_document_of_a_multi_document_manifest_is_installed(self):
        # [FUNCTION DOC] A string holding a service and a stateful set separated by a document
        # separator adds both objects, in order.
        """A caller pastes a file, and a file is `---` separated; dropping all but the first would be silent."""
        assert objects_added_by(f"{SERVICE}---\n{STATEFUL_SET}") == [
            yaml.safe_load(SERVICE),
            yaml.safe_load(STATEFUL_SET),
        ]

    def test_separate_entries_are_installed_just_as_one_joined_entry_is(self):
        # [FUNCTION DOC] Two separate entries add the same objects as one joined entry.
        """Whether a caller joins its manifests itself must not decide what the cluster gets."""
        assert objects_added_by(SERVICE, STATEFUL_SET) == objects_added_by(f"{SERVICE}---\n{STATEFUL_SET}")

    def test_the_text_the_caller_wrote_reaches_the_cluster_unchanged(self):
        # [FUNCTION DOC] The rendered output (without its trailing newlines) contains the commented
        # config map text verbatim, including its leading comment and key order.
        """The chart understands none of this text, so reserializing it could only lose something."""
        rendered = run_helm_template_run(*extra_manifests_args(COMMENTED)).stdout

        assert COMMENTED.rstrip("\n") in rendered

    def test_rendering_the_same_manifests_twice_renders_the_same_bytes(self):
        # [FUNCTION DOC] Two renders with the same two manifests produce identical standard output.
        """A relaunch is a helm upgrade, and anything freshly generated here would restart unrelated pods."""
        arguments = extra_manifests_args(SERVICE, STATEFUL_SET)

        assert run_helm_template_run(*arguments).stdout == run_helm_template_run(*arguments).stdout

    def test_a_manifest_that_is_not_text_is_refused(self):
        # [FUNCTION DOC] Passing a mapping instead of a string makes rendering fail with an error
        # mentioning the extra manifests value.
        """A yaml mapping written by mistake would be rendered as go's own formatting of it, or not at all."""
        assert "extraManifests" in render_run_error("--set-json", 'extraManifests=[{"kind":"Service"}]')

    def test_a_go_template_the_caller_quoted_stays_the_text_they_quoted(self):
        # [FUNCTION DOC] A manifest containing a release-name template expression is installed as
        # parsed with the expression text unevaluated.
        """The chart runs no tpl over this, and a caller's own braces must not become a release's name."""
        installed = objects_added_by(TEMPLATED)

        assert installed == [yaml.safe_load(TEMPLATED)]
        assert installed[0]["data"]["release"] == "{{ .Release.Name }}"

    def test_a_manifest_of_no_documents_installs_nothing(self):
        # [FUNCTION DOC] An empty string and a comment-only string both add no objects.
        """A caller that reads a file per backend hands an empty one here, and that is not a broken release."""
        assert objects_added_by("") == []
        assert objects_added_by("# the ray backend starts these itself\n") == []

    def test_the_blank_documents_around_a_manifest_are_not_objects(self):
        # [FUNCTION DOC] A service manifest surrounded by leading and trailing document separators
        # adds just the one service object.
        """A pasted file leads and trails with separators, and yaml reads each of those as a None document."""
        assert objects_added_by(f"---\n---\n{SERVICE}---\n---\n") == [yaml.safe_load(SERVICE)]


@requires_helm
class TestExtraManifestsKeepTheirIdentity:
    # [CLASS DOC] Tests that the launcher's manifest parser keeps extra objects distinct (need helm).
    def test_two_namespaces_may_hold_a_name_and_stay_two_objects(self):
        # [FUNCTION DOC] Renders with the notes config map in the run namespace and a same-named
        # one in namespace "other"; the parsed manifest's identity map has two more entries than the
        # baseline render has objects.
        """The launcher refuses a relaunch that changes an object, and folding these two would hide one."""
        rendered = run_helm_template_run(*extra_manifests_args(NOTES, ELSEWHERE)).stdout

        assert len(Manifest.parse(rendered, namespace=NAMESPACE).by_identity) == len(render_run()) + 2

    def test_two_api_groups_may_hold_a_kind_and_a_name_and_stay_two_objects(self):
        # [FUNCTION DOC] Renders with a core service and a same-named service of a custom API group;
        # the parsed identity map again has two more entries than the baseline render.
        """A custom resource can borrow a built-in's kind and name, and only apiVersion separates them."""
        rendered = run_helm_template_run(*extra_manifests_args(SERVICE, OTHER_GROUP_SERVICE)).stdout

        assert len(Manifest.parse(rendered, namespace=NAMESPACE).by_identity) == len(render_run()) + 2
