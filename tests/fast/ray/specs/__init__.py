# [PACKAGE DOC]
# Test package for the worker spec computation of the ray layer (the code under miles.ray.specs).
# Its modules check which worker specs a run produces for the inference, rollout, train and
# entrypoint specs, including deploy component filtering. The package itself defines no code; it
# exists so the test modules are importable by their dotted path.
