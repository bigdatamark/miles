# [MODULE DOC] Load-balancing utilities over sequence lengths (adapted from verl). Provides
# partitioners that split item indices into k groups with balanced total length (largest
# differencing / Karmarkar-Karp, and a greedy variant), a validated public wrapper used to balance
# across data-parallel ranks and microbatches, first-fit bin packing under a token cap (arrival
# order and decreasing order), a helper to grow a bin list by splitting bins, and an inverse-
# permutation helper. All functions are pure except the in-place bin expansion.
# Copied from https://github.com/volcengine/verl/blob/468adf22c43b744348051fccd7a5d830c6c3c36a/verl/utils/seqlen_balancing.py
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import heapq


def karmarkar_karp(seqlen_list: list[int], k_partitions: int, equal_size: bool):
    # [FUNCTION DOC] Partitions item indices into k_partitions groups whose length sums are as
    # close as possible, using the largest differencing method. Returns a list of k index lists
    # (order within a partition follows merge order, not sorted). Algorithm: sort items by
    # (length, index) ascending. If equal_size, require the item count to be divisible by k
    # (assertion error otherwise) and seed one state per consecutive block of k sorted items, each
    # state holding k sets with one item each; otherwise seed one state per item (the item in
    # one set, the other k-1 sets empty). States live in a min-heap ordered so the state with the
    # largest spread (max set sum minus min set sum) pops first, ties broken by the larger biggest
    # set. Repeatedly pop two states and merge them (set i of the first, sorted descending, absorbs
    # set k-1-i of the second, then re-sort descending) and push back until one state remains. With
    # equal_size, asserts every final partition has exactly n/k items. Nested helper classes are
    # defined inside this function on every call.
    # see: https://en.wikipedia.org/wiki/Largest_differencing_method
    class Set:
        # [CLASS DOC] HELPER: One partition under construction: a running sum of lengths and a
        # list of (index, length) pairs. Ordering compares sum first, then item count, then the
        # item list lexicographically.

        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Starts with zero sum and no items.
            self.sum = 0
            self.items = []

        def add(self, idx: int, val: int):
            # [FUNCTION DOC] HELPER: Appends the (index, length) pair and adds the length to the
            # sum.
            self.items.append((idx, val))
            self.sum += val

        def merge(self, other):
            # [FUNCTION DOC] HELPER: Appends all of the other set's items to this set and adds
            # their lengths to this sum; the other set is not modified.
            for idx, val in other.items:
                self.items.append((idx, val))
                self.sum += val

        def __lt__(self, other):
            # [FUNCTION DOC] HELPER: Strict less-than by sum, then by item count, then by
            # lexicographic comparison of the item pair lists.
            if self.sum != other.sum:
                return self.sum < other.sum
            if len(self.items) != len(other.items):
                return len(self.items) < len(other.items)
            return self.items < other.items

    class State:
        # [CLASS DOC] HELPER: A candidate k-way partitioning: k Set objects kept sorted by
        # descending sum. Defines heap ordering by spread (see the lt method).

        def __init__(self, items: list[tuple[int, int]], k: int) -> None:
            # [FUNCTION DOC] HELPER: Creates k empty sets, puts the i-th (index, length) pair
            # into the i-th set, then sorts sets descending. Asserts that the number of items is
            # either 1 or exactly k (assertion error otherwise).
            self.k = k
            # sets should always be decreasing order
            self.sets = [Set() for _ in range(k)]
            assert len(items) in [1, k], f"{len(items)} not in [1, {k}]"
            for i, (idx, seqlen) in enumerate(items):
                self.sets[i].add(idx=idx, val=seqlen)
            self.sets = sorted(self.sets, reverse=True)

        def get_partitions(self):
            # [FUNCTION DOC] HELPER: Returns, for each set in current order, the list of item
            # indices it holds (lengths dropped).
            partitions = []
            for i in range(len(self.sets)):
                cur_partition = []
                for idx, _ in self.sets[i].items:
                    cur_partition.append(idx)
                partitions.append(cur_partition)
            return partitions

        def merge(self, other):
            # [FUNCTION DOC] HELPER: Merges another state into this one in place: pairs this
            # state's i-th largest set with the other's i-th smallest set (so large pairs with
            # small), then re-sorts the sets descending by sum. The other state is not modified.
            for i in range(self.k):
                self.sets[i].merge(other.sets[self.k - 1 - i])
            self.sets = sorted(self.sets, reverse=True)

        @property
        def spread(self) -> int:
            # [FUNCTION DOC] HELPER: Difference between the largest and smallest set sums.
            return self.sets[0].sum - self.sets[-1].sum

        def __lt__(self, other):
            # [FUNCTION DOC] HELPER: Heap ordering that is reversed so the heap pops the state
            # with the largest spread first; on equal spread, the state whose largest set is
            # greater pops first.
            # least heap, let the state with largest spread to be popped first,
            # if the spread is the same, let the state who has the largest set
            # to be popped first.
            if self.spread != other.spread:
                return self.spread > other.spread
            return self.sets[0] > other.sets[0]

        def __repr__(self) -> str:
            # [FUNCTION DOC] HELPER: Debug string listing, per set, the item lengths in braces,
            # all sets in square brackets, comma-separated.
            repr_str = "["
            for i in range(self.k):
                if i > 0:
                    repr_str += ","
                repr_str += "{"
                for j, (_, seqlen) in enumerate(self.sets[i].items):
                    if j > 0:
                        repr_str += ","
                    repr_str += str(seqlen)
                repr_str += "}"
            repr_str += "]"
            return repr_str

    sorted_seqlen_list = sorted([(seqlen, i) for i, seqlen in enumerate(seqlen_list)])
    states_pq = []
    if equal_size:
        assert len(seqlen_list) % k_partitions == 0, f"{len(seqlen_list)} % {k_partitions} != 0"
        for offset in range(0, len(sorted_seqlen_list), k_partitions):
            items = []
            for i in range(k_partitions):
                seqlen, idx = sorted_seqlen_list[offset + i]
                items.append((idx, seqlen))
            heapq.heappush(states_pq, State(items=items, k=k_partitions))
    else:
        for seqlen, idx in sorted_seqlen_list:
            heapq.heappush(states_pq, State(items=[(idx, seqlen)], k=k_partitions))

    while len(states_pq) > 1:
        state0 = heapq.heappop(states_pq)
        state1 = heapq.heappop(states_pq)
        # merge states
        state0.merge(state1)
        heapq.heappush(states_pq, state0)

    final_state = states_pq[0]
    partitions = final_state.get_partitions()
    if equal_size:
        for _i, partition in enumerate(partitions):
            assert len(partition) * k_partitions == len(
                seqlen_list
            ), f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"
    return partitions


def greedy_partition(seqlen_list: list[int], k_partitions: int, equal_size: bool):
    # [FUNCTION DOC] Alternative greedy partitioner (not used by the public wrapper in this
    # module). Visits items in their original order (despite the variable name, no sorting is
    # done) and puts each into the partition with the currently smallest sum (first one on ties).
    # When equal_size, adds a bias of total length plus one to every length so that counts
    # dominate sums and partitions fill evenly, then asserts each partition has exactly n/k items.
    # Returns k lists of item indices.
    # KNOWN ISSUE: the list is named sorted_seqlen but is never sorted, so this is not the
    # classic longest-first greedy heuristic.
    bias = sum(seqlen_list) + 1 if equal_size else 0
    sorted_seqlen = [(seqlen + bias, i) for i, seqlen in enumerate(seqlen_list)]
    partitions = [[] for _ in range(k_partitions)]
    partition_sums = [0 for _ in range(k_partitions)]
    for seqlen, i in sorted_seqlen:
        min_idx = None
        for j in range(k_partitions):
            if min_idx is None or partition_sums[j] < partition_sums[min_idx]:
                min_idx = j
        partitions[min_idx].append(i)
        partition_sums[min_idx] += seqlen
    if equal_size:
        for _i, partition in enumerate(partitions):
            assert len(partition) * k_partitions == len(
                seqlen_list
            ), f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"
    return partitions


def get_seqlen_balanced_partitions(seqlen_list: list[int], k_partitions: int, equal_size: bool):
    # [FUNCTION DOC] Public entry point: balanced k-way partition of item indices by length.
    # Asserts there are at least k items, runs the largest differencing partitioner, then
    # validates and normalizes: asserts k partitions were produced, none empty, and together they
    # cover every index exactly once; returns each partition sorted ascending by index.
    # Raises assertion errors on any violation. The greedy partitioner is not used.
    """get order of seq lengths to make partitions balanced, this is
        used in balacing sum of seqlength across dp ranks and microbatches
    Parameters:
        seqlen_list (List[int]):
            seq lengths of each items
        k_partitions (int):
            resulting number of partitions
        equal_size (bool):
            if True, number of items in each partitions must be equal.
            if False, only consider balancing the sum, each partition can have
            variable number of items
    Returns:
        partitions (List[List[int]]):
            return k_partitions list containing the index of items.
    """
    assert len(seqlen_list) >= k_partitions, f"number of items:[{len(seqlen_list)}] < k_partitions:[{k_partitions}]"

    def _check_and_sort_partitions(partitions):
        # [FUNCTION DOC] HELPER: Validates the partitions as described in the enclosing function
        # (count, non-empty, full coverage via assertions) and returns a new list with each
        # partition sorted ascending.
        assert len(partitions) == k_partitions, f"{len(partitions)} != {k_partitions}"
        seen_idx = set()
        sorted_partitions = [None] * k_partitions
        for i, partition in enumerate(partitions):
            assert len(partition) > 0, f"the {i}-th partition is empty"
            for idx in partition:
                seen_idx.add(idx)
            sorted_partitions[i] = sorted(partition)
        assert seen_idx == set(range(len(seqlen_list)))
        return sorted_partitions

    partitions = karmarkar_karp(seqlen_list=seqlen_list, k_partitions=k_partitions, equal_size=equal_size)
    return _check_and_sort_partitions(partitions)


def first_fit_pack(total_lengths, max_tokens_per_bin):
    # [FUNCTION DOC] Packs item indices into bins in index order using the first-fit rule (see
    # the first-fit helper); input is a sequence of lengths and a per-bin token cap. Returns a
    # list of index lists.
    """First-fit bin packing in arrival order; bins hold indices, oversized samples land alone."""
    return _first_fit(range(len(total_lengths)), total_lengths, max_tokens_per_bin)


def first_fit_decreasing_pack(total_lengths, max_tokens_per_bin):
    # [FUNCTION DOC] Same as the arrival-order packer but visits indices sorted by length
    # descending (stable, so ties keep index order). Returns a list of index lists.
    """First-fit over indices sorted by length descending (FFD); never more bins than first-fit."""
    order = sorted(range(len(total_lengths)), key=lambda i: -total_lengths[i])
    return _first_fit(order, total_lengths, max_tokens_per_bin)


def _first_fit(order, total_lengths, max_tokens_per_bin) -> list[list[int]]:
    # [FUNCTION DOC] HELPER: Core first-fit. For each index in the given visiting order, places it
    # in the first existing bin where the bin sum plus its length does not exceed the cap
    # (inclusive), else opens a new bin. An item longer than the cap thus gets a bin alone; no
    # errors raised. Returns the bins as lists of indices in placement order.
    bins: list[list[int]] = []
    bin_sums: list[int] = []
    for idx in order:
        length = total_lengths[idx]
        for j in range(len(bins)):
            if bin_sums[j] + length <= max_tokens_per_bin:
                bins[j].append(idx)
                bin_sums[j] += length
                break
        else:
            bins.append([idx])
            bin_sums.append(length)
    return bins


def _split_bin_by_tokens(bin_indices: list[int], lengths) -> list[list[int]]:
    # [FUNCTION DOC] HELPER: Splits one bin into two lists using longest-processing-time-first:
    # visit indices by length descending, adding each to the half with the smaller (or equal,
    # preferring the first) running sum. Returns a pair of index lists.
    """Split a bin into two token-balanced halves (LPT)."""
    halves: list[list[int]] = [[], []]
    sums = [0, 0]
    for idx in sorted(bin_indices, key=lambda i: -lengths[i]):
        h = 0 if sums[0] <= sums[1] else 1
        halves[h].append(idx)
        sums[h] += lengths[idx]
    return halves


def expand_bins_by_splitting(bins: list[list[int]], target_count: int, lengths) -> None:
    # [FUNCTION DOC] Mutates the bin list in place until it has target_count bins or no bin has
    # more than one item. Each round picks, among bins with at least two items, the one with the
    # largest total length (ties go to the larger bin position), splits it with the token-balanced
    # splitter, replaces it by the first half and appends the second half at the end. Returns
    # nothing; stops silently short of the target when nothing can be split.
    """Grow ``bins`` in place to ``target_count`` by splitting the largest bins."""
    while len(bins) < target_count:
        candidates = [(sum(lengths[i] for i in b), idx) for idx, b in enumerate(bins) if len(b) > 1]
        if not candidates:
            break
        _, idx = max(candidates)
        left, right = _split_bin_by_tokens(bins[idx], lengths)
        bins[idx] = left
        bins.append(right)


def get_reverse_idx(idx_map):
    # [FUNCTION DOC] Inverts a permutation given as a sequence where position i holds the target
    # index: returns a deep copy in which entry idx_map[i] is i. Assumes idx_map is a permutation
    # of 0..n-1 and supports item assignment (e.g. a list); otherwise the result is undefined.
    reverse_idx_map = copy.deepcopy(idx_map)

    for i, idx in enumerate(idx_map):
        reverse_idx_map[idx] = i

    return reverse_idx_map
