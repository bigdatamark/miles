# [PACKAGE DOC] Auditing utilities for verifying that training and inference stay consistent.
# Subpackages: an event logger (structured, per-process event records), an event analyzer (rules
# that check recorded events, such as weight checksum agreement across replicas and inference
# engines), and witness (tracking of tensor values for cross-process comparison). Module-level
# helpers cover weight checksum normalization and process identity naming. This file defines no
# names.
