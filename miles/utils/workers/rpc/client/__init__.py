# [PACKAGE DOC]
# Client side of the worker rpc: a worker handle that exposes the remote worker's rpc methods as
# async calls (handle), the per-call state machine that submits a call and polls for its result
# (call), and the shared transport, error types and retry constants (misc). The client talks
# http to the server package of the same layer using the shared protocol in the common package.
