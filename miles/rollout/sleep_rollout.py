# [MODULE DOC]
# Placeholder rollout function that never produces data: it blocks forever, logging once per hour.
# Intended for debugging or for keeping the rollout side idle (e.g. to attach to the cluster).
import logging
import time

logger = logging.getLogger(__name__)


def sleep(args, rollout_id, data_source, evaluation=False):
    # [FUNCTION DOC] Never returns: loops forever, sleeping 3600 seconds per iteration and logging
    # an info message with the running count of hours slept. All parameters are ignored. Blocks
    # the calling thread (synchronous sleep).
    count = 0
    while True:
        time.sleep(3600)
        count += 1
        logger.info(f"rollout sleep for {count} hours")
