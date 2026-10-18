# [PACKAGE DOC]
# HELPER: Shared scenario and support code for the fault-tolerance end-to-end tests. Modules: the
# comparison application framework (baseline versus target sides, per-side configuration and
# release handling), command-line option types, comparison functions for deterministic metrics,
# execution helpers (training runner, directories, deterministic rollout arguments), the test
# mode table (model, parallel layout, cells, engines), the fault-injection subpackage (fault
# forms, event state and views), and one scenario module per test kind (random crash, fully
# async random crash, realistic GSM8K, rollout deterministic, trainer deterministic, trainer
# with and without failure). The package itself defines no code.
