# [PACKAGE DOC]
# Server side of the worker rpc: a web application (app) exposing a hosted worker object, the
# request logic (core), the executor that runs worker methods (executor) and the store that tracks
# call state and outcomes (store). Calls are submitted, run in the background and polled by call id.
