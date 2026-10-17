# [PACKAGE DOC]
# Level-triggered reconcile machinery. A source (for example a Kubernetes pod watch) streams
# events into an object store that groups objects under parent keys; a work queue with retry
# scheduling then reconciles each changed parent key at most once at a time. Modules: k8s_api
# (pod list and watch API), k8s_reflector (turns the API into source events), source_event (event
# types), source_stream_driver (feeds the store from a restartable source), object_store (cache
# grouped by parent), work_queue and retry_scheduler (deduplicated work with backoff), and loop
# (the loop tying all of them together).
