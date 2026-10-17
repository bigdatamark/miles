# [PACKAGE DOC]
# Tinker-compatible gateway for Miles: lets clients built with the Tinker training SDK drive
# LoRA training and sampling on a Miles cluster. Subpackages and modules: core (transport-
# independent service logic: sessions, adapter-slot models, ordered command queues, scheduling,
# futures, checkpoints), server (HTTP application translating the SDK's JSON and protobuf wire
# formats), runtime (backend adapting the service to the Miles trainer and the SGLang router) and
# arguments (launch flags and compatibility checks). The package init itself defines nothing.
