# [PACKAGE DOC]
# Cell registration: lets deployments that run elsewhere (their own pods or processes) announce
# the cells they host to the run, which serves them through a hub acting as a worker provider.
# Reporters (inside deployments) observe their own cells and periodically send snapshots to the
# hub; the hub merges the snapshots, expires silent reporters, and exposes the cells to watchers.
# Modules: models (snapshot wire models), reporter (sending side), hub (receiving side).
