# [PACKAGE DOC] Colocate pairing controller. When inference workers share GPUs with trainer
# workers (colocation), each inference pod must land on the node of one specific trainer pod and
# use specific GPUs of it. Inference pods are created held back by a scheduling gate; the
# controller watches pods of a release, and once the paired trainer pod has a node, pins the
# inference pod to that node, records its base GPU id in an annotation and removes the gate.
# Modules: config (validated layout description), pods (pod coordinates and patch building),
# controller (placement computation and reconcile logic), __main__ (process entry point).
