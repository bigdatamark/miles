# [MODULE DOC]
# HELPER: Minimal fakes of a ray remote method and of the rollout executor actor for tests that only
# need to see which remote calls were made.
class FakeRemoteMethod:
    # [CLASS DOC] HELPER: Fake of a ray actor method handle exposing a remote call that records its
    # positional arguments and returns a fake object reference string.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Initializes the empty list of recorded argument tuples.
        self.calls: list[tuple] = []

    def remote(self, *args) -> str:
        # [FUNCTION DOC] HELPER: Records the positional arguments (keyword arguments are not accepted)
        # and returns a string object-ref-N where N is the number of calls so far, starting at 1.
        self.calls.append(args)
        return f"object-ref-{len(self.calls)}"


class FakeRolloutExecutor:
    # [CLASS DOC] HELPER: Fake rollout executor with only the set-train-parallel-config remote method,
    # backed by a recording fake.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the recording fake for the set-train-parallel-config method.
        self.set_train_parallel_config = FakeRemoteMethod()
