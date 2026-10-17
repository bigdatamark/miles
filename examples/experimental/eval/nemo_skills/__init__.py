# [PACKAGE DOC]
# Delegate implementation that evaluates a model through NeMo Skills. It has three modules: a
# configuration module with structured dataset and environment records, a client that posts an
# evaluation request (router address plus benchmark parameters) to a Skills evaluation server over
# HTTP, and a standalone Flask server that runs the NeMo Skills evaluation command line tool for
# each requested benchmark against the training run's inference router and returns the collected
# metrics. Importing the package has no side effects.
"""NeMo Skills evaluation helpers."""
