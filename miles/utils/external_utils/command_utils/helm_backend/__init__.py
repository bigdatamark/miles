# [PACKAGE DOC] Kubernetes implementation of the command backend, driven through helm and
# kubectl. backend defines the backend class (training launches and one-off commands run as
# jobs), command_job renders, runs and monitors a one-off multi-completion job from the chart's
# command-job template, and the launcher subpackage holds the training-launch machinery (helm
# and kubectl wrappers, values, hot restart, observability). This package file itself defines
# nothing.
