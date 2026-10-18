# [MODULE DOC] HELPER: Fixture module for the rpc metadata tests. It deliberately turns on postponed
# evaluation of annotations (so every annotation is stored as a string) and defines a strict payload
# model plus a worker class using it, to check that annotation strings are resolved in the worker's
# own module namespace. It must stay in its own module so the postponed behavior is not shared with
# the test file.
from __future__ import annotations

from miles.utils.pydantic_utils import StrictBaseModel


class LatePayload(StrictBaseModel):
    # [CLASS DOC] HELPER: Strict pydantic model with one required text field; used as both the
    # parameter type and the return type of the worker method, so only string annotations refer to it.
    text: str


class PostponedWorker:
    # [CLASS DOC] HELPER: Worker with a single public method whose annotations are postponed strings.
    def demo_transform(self, payload: LatePayload) -> LatePayload:
        # [FUNCTION DOC] HELPER: Returns the given payload unchanged.
        return payload
