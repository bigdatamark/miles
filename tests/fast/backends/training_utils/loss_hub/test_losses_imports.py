# [MODULE DOC] CPU test of the import weight of the loss hub: importing its losses module must not
# pull in ray or the miles misc utilities (checked via an import-isolation helper).
from tests.fast.import_isolation_utils import modules_imported_by


class TestLossHubImportWeight:
    # [CLASS DOC] Import-weight tests for the loss hub.
    def test_losses_import_does_not_require_ray(self):
        # [FUNCTION DOC] Test: the set of modules imported by the losses module contains neither ray
        # nor the miles misc utilities module.
        """The loss hub resolves custom losses through function_registry, so importing it must not pull in ray."""
        modules = modules_imported_by("miles.backends.training_utils.loss.hub.losses")

        assert "ray" not in modules
        assert "miles.utils.misc" not in modules
