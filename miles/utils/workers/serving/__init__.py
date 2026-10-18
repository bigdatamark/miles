# [PACKAGE DOC]
# Process-level serving of workers over rpc. serve is the outer launcher that computes the worker
# environment and replaces itself with serve_inner, which builds the worker and runs the rpc
# server. serve_actor hosts a worker inside a Ray actor. utils holds argument and socket helpers;
# worker_identity derives a worker's indices and gpu ids from platform environment variables.
