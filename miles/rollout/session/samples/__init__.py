# [PACKAGE DOC]
# Sample handling for the session server. Contains the merge module, which turns the per-turn
# session records into training samples on the server (compute per-turn samples aligned with the
# token-in token-out accumulated tokens, truncate by total token budget, merge turns, optionally
# assemble additional routing-replay patches), and the codec module, which defines the wire
# format (safetensors with a JSON metadata tensor) used to ship the computed sample fields from the
# server to the driver and to overlay them onto the driver's input sample. The package init
# itself defines nothing.
