# [PACKAGE DOC]
# Orchestrator side of the helm backend: code that runs inside the orchestrator pod (a wrapper that
# runs the orchestration script and publishes its exit code through a state file) and the launcher's
# view of it (state file model and the observer that waits for the run verdict).
