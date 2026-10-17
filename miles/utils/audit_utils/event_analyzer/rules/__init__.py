# [PACKAGE DOC] Analysis rules for audit events. Each submodule exposes a check function that
# takes the list of events and returns a list of issue records (empty when all is well). Rules:
# cross-replica weight checksum, inference-engine weight checksum consistency and witness data
# accounting; checksum_compare holds the shared comparison helpers. This file defines no names.
