# [PACKAGE DOC] Training launcher for the Kubernetes backend. entrypoint implements the launch
# (build arguments and values, plan hot restart, check upgrades, helm upgrade, follow the run);
# command_wrapper wraps the helm and kubectl command lines; hot_restart plans which components
# restart in place; launch_record fingerprints what a launch installed; manifest_types and
# manifest_diff parse rendered manifests and classify changes between installs; observability
# streams pod logs and collects failure diagnosis; values (a subpackage) builds the chart values.
# This package file itself defines nothing.
