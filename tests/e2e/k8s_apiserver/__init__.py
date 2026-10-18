# [PACKAGE DOC] End-to-end tests of the Kubernetes reconcile/reflector code against a real
# Kubernetes API server (plus etcd) started in docker containers, with no kubelet or nodes. The
# package provides the container lifecycle helpers (apiserver), environment/docker detection
# helpers (environment), generic client and wait helpers (utils), reflector test helpers
# (reflector_utils), pytest fixtures (conftest), and the test modules themselves. Tests skip when
# docker is unavailable unless the environment demands it.
