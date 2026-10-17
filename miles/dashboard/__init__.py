# [PACKAGE DOC]
# Miles run dashboard: lightweight telemetry collection and viewing for a training run. A collector
# actor (collector) receives metrics, phase events and trajectory records from all processes and
# persists them to a dashboard directory under the dump-details directory; GPU utilization samplers
# (gpu_sampler) and an engine metrics scraper (sglang_scraper) feed it; hooks wire timers and the
# router into it; backend holds the process-level lifecycle and logging entry points; args defines the
# command-line options and builds the collector configuration; store/dump_reader read the persisted
# data back; advisory derives heuristic health alarms and tuning suggestions; logging_utils has a
# warning rate limiter. The package file exports nothing; telemetry failures must never break a
# training run.
