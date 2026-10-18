# [PACKAGE DOC]
# Helm-chart specific knowledge for Kubernetes runs: builder derives the Kubernetes backend
# capability for a run deployed by the chart; env reads the release and namespace and holds the label
# keys the chart uses; naming reproduces the chart's object naming rules (name length budgets, digest
# shortening, static host names).
