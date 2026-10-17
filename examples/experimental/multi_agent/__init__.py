# [PACKAGE DOC]
# Example package for a multi-agent math-reasoning rollout: for each problem a solver stage produces
# several candidate solutions in parallel, a rewriter stage revises them using all candidates, and a
# selector stage picks the best revision; rewards are re-weighted by whether the final selected
# answer is correct. Modules: prompt templates, the agent classes and pipeline, and the generate
# function that plugs the pipeline into rollout. The package defines no names itself.
