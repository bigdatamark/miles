# [PACKAGE DOC]
# Fast tests for the hot-restart deploy scenario support code: fact collectors describing the
# cluster and restart behaviour (cluster facts, restart facts), the assertions that judge a
# restart (process, redone-from-checkpoint, redone-from-scratch, workloads), the cluster
# observer, the driver, evidence gathering, the fault form, freeze plans and the scenario
# definitions themselves. All tests use fakes instead of a real cluster. The package defines no
# names itself; the two fact modules in this directory are test helper modules, not tests.
