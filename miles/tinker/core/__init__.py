# [PACKAGE DOC]
# Transport-independent core of the Tinker gateway. Modules: types (commands, records, gateway
# configuration, errors), input_validation (checks of client-supplied content), future (in-memory
# request futures for long-poll retrieval), model_queue (per-model ordered command queues),
# scheduler (cross-model selection and packing of work units), utils (checkpoint path and metadata
# helpers), and service (sessions, models, sampling, lease expiry and the dispatcher loop driving a
# trainer backend). The package init itself defines nothing.
