# [PACKAGE DOC] Worker-side infrastructure shared by cluster backends: port allocation across nodes,
# command-line argument rendering/parsing helpers for launching workers, and backend capability
# declarations. The package init is empty and re-exports nothing; the entrypoint of light workers
# keeps its import footprint small, so modules here avoid heavy imports.
