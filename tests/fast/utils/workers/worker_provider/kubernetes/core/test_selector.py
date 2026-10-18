# [MODULE DOC] Tests for the provider's pod watch label selector builder: with no wanted pools it
# appends a label no pod carries (watch nothing); with pools it appends "<pool label key> in
# (<sorted comma-separated pool ids>)"; the base selector always comes first.
from __future__ import annotations

from miles.utils.workers.worker_provider.kubernetes.core import provider
from miles.utils.workers.worker_provider.kubernetes.helm import env

BASE_SELECTOR = "app.kubernetes.io/instance=r"


class TestWatchedPodsSelector:
    # [CLASS DOC] Tests of the selector text.
    def test_watches_nothing_when_no_pool_is_wanted(self) -> None:
        # [FUNCTION DOC] Empty pool list: selector is the base, a comma, and the provider's
        # never-matching label constant.
        """A run with no dynamic pool must watch zero pods, so the selector names a label no pod carries."""
        selector = provider._watched_pods_selector(
            base_selector=BASE_SELECTOR, pool_label_key=env.DEFAULT_LABEL_KEYS.pool_id, pool_ids=[]
        )

        assert selector == f"{BASE_SELECTOR},{provider._NO_POD_CARRIES_THIS_LABEL}"

    def test_scopes_the_watch_to_the_wanted_pools(self) -> None:
        # [FUNCTION DOC] One pool "engine": selector is base plus ",<pool key> in (engine)".
        """Pods of other pools in the same release must stay outside the watch."""
        selector = provider._watched_pods_selector(
            base_selector=BASE_SELECTOR, pool_label_key=env.DEFAULT_LABEL_KEYS.pool_id, pool_ids=["engine"]
        )

        assert selector == f"{BASE_SELECTOR},{env.DEFAULT_LABEL_KEYS.pool_id} in (engine)"

    def test_sorts_the_wanted_pools(self) -> None:
        # [FUNCTION DOC] Pools given as rollout, engine are rendered sorted: "(engine,rollout)".
        """The selector is compared and logged, so the same pool set must always spell the same string."""
        selector = provider._watched_pods_selector(
            base_selector=BASE_SELECTOR,
            pool_label_key=env.DEFAULT_LABEL_KEYS.pool_id,
            pool_ids=["rollout", "engine"],
        )

        assert selector == f"{BASE_SELECTOR},{env.DEFAULT_LABEL_KEYS.pool_id} in (engine,rollout)"

    def test_keeps_the_base_selector_in_front(self) -> None:
        # [FUNCTION DOC] The result starts with the base selector "app=miles," for a custom base.
        """The release selector is what keeps one run from watching another run's pods."""
        selector = provider._watched_pods_selector(
            base_selector="app=miles", pool_label_key="pool", pool_ids=["engine"]
        )

        assert selector.startswith("app=miles,")
