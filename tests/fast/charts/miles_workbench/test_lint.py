# [MODULE DOC]
# Tests (need helm) that helm lint passes on the workbench chart with default values and with
# several supported value combinations (a claim-backed volume, no volumes, no leader-worker-set
# rights, and no rbac with a pre-existing service account).
import pytest
from tests.fast.charts.utils import requires_helm, run_helm_lint


@requires_helm
class TestChartLint:
    # [CLASS DOC] Lint tests of the workbench chart.
    def test_lint_passes_on_the_defaults(self):
        # [FUNCTION DOC] Lint with no extra values exits zero.
        """The chart installs with no values at all beyond the cluster's own."""
        result = run_helm_lint()
        assert result.returncode == 0, result.stdout + result.stderr

    @pytest.mark.parametrize(
        "overrides",
        [
            [
                "--set-json",
                'infra.volumes=[{"name":"cluster-storage","persistentVolumeClaim":{"claimName":"shared"},'
                '"mounts":[{"mountPath":"/cluster-storage"}]}]',
            ],
            ["--set-json", "infra.volumes=[]"],
            ["--set", "rbac.leaderWorkerSets=false"],
            ["--set", "rbac.create=false", "--set", "serviceAccount.name=preexisting"],
        ],
    )
    def test_lint_passes_for_every_supported_combination(self, overrides):
        # [FUNCTION DOC] Parametrized over four override argument lists; lint with each exits zero.
        """Linting renders NOTES.txt too, so every branch of every template has to hold up."""
        result = run_helm_lint(*overrides)

        assert result.returncode == 0, result.stdout + result.stderr
