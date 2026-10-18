# [MODULE DOC] Tests for the helm object-name budget: the reserved cell index suffix must fit the
# widest padded index, the component name budget plus both suffixes (cell index, revision hash)
# must equal the object name limit, component_name keeps short names readable and bounds long
# ones, and the static worker host of the widest cell index has every dot-separated label within
# the limit.
from miles.utils.workers.naming import NAME_INDEX_PAD_WIDTH
from miles.utils.workers.worker_provider.kubernetes.helm.naming import (
    COMPONENT_NAME_BUDGET,
    LONGEST_CELL_INDEX_SUFFIX,
    LONGEST_REVISION_HASH_SUFFIX,
    MAX_OBJECT_NAME_LENGTH,
    component_name,
    static_worker_host,
)

_LONGEST_CELL_INDEX = 10**NAME_INDEX_PAD_WIDTH - 1


class TestTheNameBudget:
    # [CLASS DOC] Tests of the constants of the name budget.
    def test_the_reserved_cell_suffix_covers_the_longest_ordinal_a_name_index_can_carry(self):
        # [FUNCTION DOC] The longest cell index suffix equals a dash followed by the largest number
        # with the padded width (all nines).
        """num_cells is an unbounded int, so the budget has to reserve room for the widest index that is padded."""
        assert LONGEST_CELL_INDEX_SUFFIX == f"-{_LONGEST_CELL_INDEX}"

    def test_the_budget_leaves_room_for_both_suffixes_inside_the_object_name_limit(self):
        # [FUNCTION DOC] Component budget plus the lengths of both longest suffixes equals the
        # object name limit exactly.
        """A descendant name is the component name plus both suffixes, and kubernetes rejects it past 63 characters."""
        longest = COMPONENT_NAME_BUDGET + len(LONGEST_CELL_INDEX_SUFFIX) + len(LONGEST_REVISION_HASH_SUFFIX)

        assert longest == MAX_OBJECT_NAME_LENGTH


class TestComponentName:
    # [CLASS DOC] Tests of component and host name construction.
    def test_a_run_of_a_hundred_thousand_cells_still_names_its_pods_within_the_limit(self):
        # [FUNCTION DOC] An 80-character release and 80-character component give a name within the
        # component budget, and with both longest suffixes within the object name limit.
        """A run past a thousand cells used to render LWS descendants kubernetes refuses outright."""
        name = component_name("a" * 80, "b" * 80)

        assert len(name) <= COMPONENT_NAME_BUDGET
        assert len(f"{name}{LONGEST_CELL_INDEX_SUFFIX}{LONGEST_REVISION_HASH_SUFFIX}") <= MAX_OBJECT_NAME_LENGTH

    def test_a_short_release_and_component_are_left_spelled_out(self):
        # [FUNCTION DOC] Release "myrun" and component "trainer" give "myrun-miles-run-trainer".
        """Reserving more budget must not start hashing the names a readable run already had."""
        assert component_name("myrun", "trainer") == "myrun-miles-run-trainer"

    def test_the_host_of_the_last_cell_of_such_a_run_is_a_legal_dns_label(self):
        # [FUNCTION DOC] The static worker host for 80-character names and the widest cell index
        # has every dot-separated label at most the object name limit long.
        """The pod hostname is the component name and the ordinal, and every label of it is capped at 63."""
        host = static_worker_host("a" * 80, "b" * 80, _LONGEST_CELL_INDEX)

        assert all(len(label) <= MAX_OBJECT_NAME_LENGTH for label in host.split("."))
