# [PACKAGE DOC]
# HELPER: Code for the hot-restart deploy tests, where a live deployed training run has the
# orchestration script and rollout executor workloads replaced (by upgrading its helm release with
# a hot-restart argument) while the trainers keep running. Modules: the cluster observer (snapshots
# of pods, workloads and trainer boot uuid), the evidence data model, the driver (waits for the run
# to freeze at scheduled steps, then relaunches), the fault form (the same take-over as a random
# fault injection), the freeze plan files, the assertion modules (observation quality, which
# workloads were rolled, which steps were redone from a checkpoint or from scratch) and the two
# scenarios (deterministic baseline-versus-target comparison, realistic GSM8K soak). The package
# itself defines no code.
