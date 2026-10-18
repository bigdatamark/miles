# [MODULE DOC]
# Tests for the split deployment example's run address book: an object holding run id, run uuid
# and namespace that names helm releases per deployment component (and optional instance id) and
# builds the argument values by which the parts of a split run find each other (the trainer
# controller addresses, the inference controller address, the shared object store arguments).
# Covered: building from a train config (run uuid and namespace are required, with the assertion
# messages), distinct release names containing the run id, trainer addresses carrying the
# release, pool id and namespace (one per trainer), the inference controller living in the primary
# release, and the shared object store master living in the primary release.
import pytest
from examples.infra_features.split_deployment.address_book import DEFAULT_TRAINER_ID, RunAddressBook

from miles.ray.specs.inference import INFERENCE_CONTROLLER_POOL_ID
from miles.ray.specs.train import compute_trainer_controller_pool_id
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.workers.types import ClusterBackend, DeployComponent
from miles.utils.workers.worker_provider.kubernetes.helm.naming import static_worker_host

NAMESPACE: str = "rl"
RUN_ID: str = "demo"
RUN_UUID: str = "0123456789abcdef"


@pytest.fixture
def address_book() -> RunAddressBook:
    # [FUNCTION DOC] HELPER: Fixture returning an address book for the module run id, run uuid and namespace.
    return RunAddressBook(run_id=RUN_ID, run_uuid=RUN_UUID, namespace=NAMESPACE)


class TestOfConfig:
    # [CLASS DOC] Tests for building the address book from a train config.
    def test_the_run_uuid_every_command_was_given_is_the_one_the_run_is_installed_under(self):
        # [FUNCTION DOC] A Kubernetes config with namespace, run id and run uuid yields an address book carrying that run uuid.
        """A launch that invented one would install a release nothing else of this run recognises."""
        config = ExecuteTrainConfig(
            cluster_backend=ClusterBackend.KUBERNETES, namespace=NAMESPACE, run_id=RUN_ID, run_uuid=RUN_UUID
        )

        assert RunAddressBook.of_config(config).run_uuid == RUN_UUID

    def test_a_command_carrying_no_run_uuid_is_refused(self):
        # [FUNCTION DOC] A config with no run uuid raises an assertion error saying every command needs the same run uuid option.
        """miles hands a split launch no run uuid of its own, so the parts would join nothing at all."""
        config = ExecuteTrainConfig(cluster_backend=ClusterBackend.KUBERNETES, namespace=NAMESPACE, run_id=RUN_ID)

        with pytest.raises(AssertionError, match="the same --run-uuid"):
            RunAddressBook.of_config(config)

    def test_a_command_carrying_no_namespace_is_refused(self):
        # [FUNCTION DOC] A config with no namespace raises an assertion error saying a namespace is half of every such name.
        """Half of every in-cluster name is the namespace, so the addresses would resolve to nothing."""
        config = ExecuteTrainConfig(cluster_backend=ClusterBackend.KUBERNETES, run_id=RUN_ID, run_uuid=RUN_UUID)

        with pytest.raises(AssertionError, match="a namespace is half of every such name"):
            RunAddressBook.of_config(config)


class TestRunAddressBook:
    # [CLASS DOC] Tests for release names and the address arguments of the address book.
    def test_every_deployment_of_a_run_installs_a_release_of_its_own(self, address_book):
        # [FUNCTION DOC] Builds release names for primary, trainer, and inference instances a and b;
        # asserts all four are distinct and each contains the run id.
        """The deployments share a run id, so only the component and instance tell their releases apart."""
        releases = [
            address_book.release(DeployComponent.PRIMARY),
            address_book.release(DeployComponent.TRAINER),
            address_book.release(DeployComponent.INFERENCE, "a"),
            address_book.release(DeployComponent.INFERENCE, "b"),
        ]

        assert len(set(releases)) == len(releases)
        assert all(RUN_ID in release for release in releases)

    def test_the_trainer_controller_is_reached_in_the_release_that_carries_it(self, address_book):
        # [FUNCTION DOC] For the default trainer id with no instance id, the trainer controller
        # addresses argument contains the trainer id followed by an equals sign, the static worker
        # host built from the trainer release, the trainer controller pool id and cell index zero,
        # the pool id, and the namespace.
        """The orchestration script dials the trainer, which a different release installed."""
        arg = address_book.trainer_controller_addrs_arg(deploy_instance_id_of_trainer_id={DEFAULT_TRAINER_ID: None})

        assert f"{DEFAULT_TRAINER_ID}=" in arg
        assert (
            static_worker_host(
                release=address_book.release(DeployComponent.TRAINER),
                component=compute_trainer_controller_pool_id(DEFAULT_TRAINER_ID),
                cell_index=0,
            )
            in arg
        )
        assert compute_trainer_controller_pool_id(DEFAULT_TRAINER_ID) in arg
        assert NAMESPACE in arg

    def test_every_trainer_of_a_run_is_reached_in_the_release_that_carries_that_one(self, address_book):
        # [FUNCTION DOC] With two trainers a and b, each with its own instance id, asserts the
        # addresses argument contains each trainer's release name and controller pool id.
        """A run training several policies deploys a trainer per policy, and one address per trainer reaches them."""
        arg = address_book.trainer_controller_addrs_arg(deploy_instance_id_of_trainer_id={"a": "a", "b": "b"})

        assert address_book.release(DeployComponent.TRAINER, "a") in arg
        assert address_book.release(DeployComponent.TRAINER, "b") in arg
        assert compute_trainer_controller_pool_id("a") in arg
        assert compute_trainer_controller_pool_id("b") in arg

    def test_the_inference_controller_is_reached_in_the_release_that_carries_it(self, address_book):
        # [FUNCTION DOC] Asserts the inference controller address argument contains the primary
        # release name, the inference controller pool id and the namespace.
        """An engine deployment registers into the one controller, which the driving release installed."""
        arg = address_book.inference_controller_addr_arg()

        assert address_book.release(DeployComponent.PRIMARY) in arg
        assert INFERENCE_CONTROLLER_POOL_ID in arg
        assert NAMESPACE in arg

    def test_an_engine_deployment_is_not_told_to_dial_itself(self, address_book):
        # [FUNCTION DOC] Asserts the inference controller address argument contains the primary
        # release but none of the trainer or inference releases.
        """Pointing an engine deployment at its own release would leave the run with no controller at all."""
        arg = address_book.inference_controller_addr_arg()
        elsewhere = [
            address_book.release(DeployComponent.TRAINER),
            address_book.release(DeployComponent.INFERENCE, "a"),
            address_book.release(DeployComponent.INFERENCE, "b"),
        ]

        assert address_book.release(DeployComponent.PRIMARY) in arg
        assert not [release for release in elsewhere if release in arg]

    def test_every_deployment_redeems_its_references_at_one_object_store(self, address_book):
        # [FUNCTION DOC] Asserts the shared object store arguments contain the primary release name and the namespace.
        """The object store master runs beside the orchestration script, and the others dial it there."""
        args = address_book.shared_object_store_args()

        assert address_book.release(DeployComponent.PRIMARY) in args
        assert NAMESPACE in args
