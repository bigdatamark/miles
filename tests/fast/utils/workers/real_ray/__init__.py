# [PACKAGE DOC]
# Worker tests that run against a real local Ray runtime (rather than the in-process fake): the
# worker manager actor, its comm conformance, the serve actor rpc and worker handles. Shared
# fixtures (probe workers, manager launchers, placement groups) live in the sibling conftest.
