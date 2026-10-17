# [PACKAGE DOC]
# Fault tolerance http api server. Exposes cells in a kubernetes-like resource style (api version
# and kind fields, patch to suspend or resume, status conditions) over a fastapi application run
# by uvicorn on a background thread. Modules: models (wire types), handles (per cell type access to
# operations and statuses), registry (routing a cell id to its handler), server (app and startup).
