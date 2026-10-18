# [PACKAGE DOC] End-to-end tests that run against a real Kubernetes cluster created with kind
# (Kubernetes in docker), or against an existing cluster named by a kubeconfig environment
# variable. Provides kind binary resolution and cluster lifecycle helpers (kind_cluster), a
# kubeconfig-based client builder (utils), pytest fixtures (conftest), and the tests: pod
# reflector behavior and environment checks against a real cluster plus launcher observability.
# Reuses the run id, docker check and namespace helpers of the API server test package.
