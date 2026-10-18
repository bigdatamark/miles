# [MODULE DOC] Unit tests for the worker naming helpers: zero-padded five digit name indices, port
# names shortened to the Kubernetes service name limit, cell ids (pool id plus padded cell index),
# worker names (pool id plus padded cell index and in-cell index), their parsers (only the last two
# hyphens separate indices, so pool ids may contain hyphens), and the cell id of a worker.
import pytest

from miles.utils.workers.naming import (
    NAME_INDEX_PAD_WIDTH,
    _worker_name_of_cell,
    cell_id_of_worker,
    compute_cell_id,
    compute_port_name,
    compute_worker_name,
    format_name_index,
    parse_cell_id,
    parse_worker_name,
)


class TestFormatNameIndex:
    # [CLASS DOC] Tests the fixed-width index formatter (valid range 0 to 99999).
    def test_pads_to_the_declared_width(self):
        # [FUNCTION DOC] Index 3 becomes "00003" and index 0 has the declared pad width in characters.
        """Every index is rendered at the same width, so names sort lexicographically by index."""
        assert format_name_index(3) == "00003"
        assert len(format_name_index(0)) == NAME_INDEX_PAD_WIDTH

    def test_accepts_the_largest_five_digit_index(self):
        # [FUNCTION DOC] Index 99999 renders as "99999".
        """The largest representable index must retain the fixed-width naming contract."""
        assert format_name_index(99999) == "99999"

    @pytest.mark.parametrize("index", [-1, 100000])
    def test_rejects_an_index_outside_the_five_digit_range(self, index: int):
        # [FUNCTION DOC] Parametrized over minus one and 100000; each raises an assertion error saying
        # the name index must be in range.
        """Every emitted index must fit the fixed-width range used for sortable names."""
        with pytest.raises(AssertionError, match="name index must be"):
            format_name_index(index)


class TestComputePortName:
    # [CLASS DOC] Tests derivation of a port name valid as a Kubernetes port name.
    def test_shortens_a_long_underscored_name_without_a_trailing_dash(self):
        # [FUNCTION DOC] The 24-character name with an underscore becomes "disaggregation" (cut to
        # fifteen characters, underscore turned into a dash, and no trailing dash left).
        """disaggregation_bootstrap is the name that first hit the fifteen-character cut."""
        assert compute_port_name("disaggregation_bootstrap") == "disaggregation"

    @pytest.mark.parametrize("name", ["9000", "80_8080"])
    def test_rejects_a_name_of_digits_alone(self, name: str):
        # [FUNCTION DOC] Parametrized over a purely numeric name and a digits-and-underscore name;
        # each raises an assertion error saying Kubernetes rejects it.
        """An IANA service name carries at least one letter, so a port named by its number is not one."""
        with pytest.raises(AssertionError, match="which Kubernetes rejects"):
            compute_port_name(name)


class TestComputeCellId:
    # [CLASS DOC] Tests cell id construction.
    def test_pads_the_cell_index(self):
        # [FUNCTION DOC] Pool "trainer-engine-actor" with cell 3 gives "trainer-engine-actor-00003".
        """A cell id carries a padded index so a pool's cell ids sort in index order."""
        assert compute_cell_id(pool_id="trainer-engine-actor", cell_index=3) == "trainer-engine-actor-00003"

    def test_orders_cell_ids_lexicographically(self):
        # [FUNCTION DOC] Cell ids for indices 10, 2 and 1 sort into the order of indices 1, 2, 10.
        """Sorted cell ids of one pool must run 0, 1, ..., 10 rather than 0, 1, 10, 2."""
        cell_ids = [compute_cell_id(pool_id="engine", cell_index=index) for index in [10, 2, 1]]

        assert sorted(cell_ids) == [compute_cell_id(pool_id="engine", cell_index=index) for index in [1, 2, 10]]


class TestComputeWorkerName:
    # [CLASS DOC] Tests worker name construction.
    def test_pads_both_indices(self):
        # [FUNCTION DOC] Pool "engine", cell 3, worker 7 gives "engine-00003-00007".
        """A worker name pads the cell index and the in-cell index alike."""
        assert compute_worker_name(pool_id="engine", cell_index=3, worker_in_cell_index=7) == "engine-00003-00007"

    def test_defaults_to_the_single_worker_of_the_single_cell(self):
        # [FUNCTION DOC] With only a pool id, both indices default to zero.
        """A one-worker pool still gets padded indices from the defaults."""
        assert compute_worker_name(pool_id="rollout-executor") == "rollout-executor-00000-00000"


class TestParseCellId:
    # [CLASS DOC] Tests cell id parsing.
    def test_round_trips_a_computed_cell_id(self):
        # [FUNCTION DOC] A computed cell id for pool "trainer-pool-a" and cell 12 parses back to that
        # pool id and the integer 12.
        """A cell id produced here parses back to the values it was built from."""
        cell_id = compute_cell_id(pool_id="trainer-pool-a", cell_index=12)

        assert parse_cell_id(cell_id) == ("trainer-pool-a", 12)


class TestParseWorkerName:
    # [CLASS DOC] Tests worker name parsing into pool id, cell index and worker index.
    def test_preserves_a_hyphenated_pool_id(self):
        # [FUNCTION DOC] "trainer-pool-a-00003-00007" parses to pool "trainer-pool-a", 3 and 7.
        """Only the last two hyphens separate indices, so a hyphenated pool id survives intact."""
        assert parse_worker_name("trainer-pool-a-00003-00007") == ("trainer-pool-a", 3, 7)

    def test_returns_integer_indices(self):
        # [FUNCTION DOC] "rollout-00012-00005" parses to pool "rollout" and the integers 12 and 5.
        """Both index suffixes come back as integers, not as the strings they were parsed from."""
        pool_id, cell_index, worker_in_cell_index = parse_worker_name("rollout-00012-00005")

        assert pool_id == "rollout"
        assert (cell_index, worker_in_cell_index) == (12, 5)
        assert isinstance(cell_index, int) and isinstance(worker_in_cell_index, int)

    def test_round_trips_a_computed_worker_name(self):
        # [FUNCTION DOC] A computed name parses back to its inputs.
        """A name produced by compute_worker_name parses back to the values it was built from."""
        worker_name = compute_worker_name(pool_id="trainer-pool-a", cell_index=2, worker_in_cell_index=4)

        assert parse_worker_name(worker_name) == ("trainer-pool-a", 2, 4)

    @pytest.mark.parametrize("worker_name", ["trainer", "trainer-1"])
    def test_rejects_missing_index_suffixes(self, worker_name: str):
        # [FUNCTION DOC] Names with no or only one index suffix raise a value error.
        """A name without both index suffixes cannot be unpacked into pool, cell, and worker."""
        with pytest.raises(ValueError):
            parse_worker_name(worker_name)

    @pytest.mark.parametrize("worker_name", ["trainer-abc-7", "trainer-3-abc"])
    def test_rejects_non_numeric_index_suffixes(self, worker_name: str):
        # [FUNCTION DOC] Names whose cell or worker suffix is not a number raise a value error.
        """A non-numeric cell or worker suffix is rejected instead of flowing on as a string."""
        with pytest.raises(ValueError):
            parse_worker_name(worker_name)


class TestCellIdOfWorker:
    # [CLASS DOC] Tests mapping a worker name to the id of its cell.
    def test_names_the_cell_the_worker_belongs_to(self):
        # [FUNCTION DOC] "engine-00003-00007" maps to "engine-00003" (the last index is dropped).
        """The cell of a worker is the padded cell id, whatever the worker name's own padding."""
        assert cell_id_of_worker("engine-00003-00007") == "engine-00003"


class TestWorkerNameOfCell:
    # [CLASS DOC] Tests the private helper naming a worker of a cell.
    def test_round_trips_with_cell_id_of_worker(self):
        # [FUNCTION DOC] Naming worker 7 of a computed cell id and mapping the name back to its cell
        # id returns the original cell id.
        """Naming a worker of a cell and asking which cell it belongs to are inverses."""
        cell_id = compute_cell_id(pool_id="engine", cell_index=3)

        assert cell_id_of_worker(_worker_name_of_cell(cell_id, worker_in_cell_index=7)) == cell_id
