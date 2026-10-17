# [PACKAGE DOC] Message-level equivalence policies used to decide whether a message replayed by a
# client equals the stored one in a session. Two modules: utils (matcher type alias, constants
# and normalization helpers) and funcs (the three built-in matchers named strict,
# loose_tool_call and role_content_only, the selector resolver that also accepts a dotted
# import path, a runtime contract wrapper and the append-only history validator). This file
# re-exports the public names; the package depends only on the standard library at import time.

"""Session message matching policies and selector resolution.

Owns every message-level equivalence policy the session server can select
via ``--session-message-matcher``, plus the strict append-only validation
that shares ``_TEMPLATE_RELEVANT_KEYS`` with the matchers.  Template
loading and rendering stay in ``template.py``; this package must depend
only on the standard library (``load_function`` is imported lazily inside
the resolver so plain matcher use never pulls in Ray via
``miles.utils.misc``).

Layout: ``utils`` holds the shared type alias, constants, and
representation-level normalization helpers that custom matchers can reuse;
``funcs`` holds the built-in matchers, the selector resolver, and the
append-only validation.
"""

from miles.utils.chat_template_utils.message_matcher_hub.funcs import (
    SessionMessageMatcherError,
    assert_messages_append_only_with_allowed_role,
    loose_tool_call_message_matches,
    resolve_session_message_matcher,
    role_content_only_message_matches,
    strict_message_matches,
)
from miles.utils.chat_template_utils.message_matcher_hub.utils import SessionMessageMatcher

__all__ = [
    "SessionMessageMatcher",
    "SessionMessageMatcherError",
    "assert_messages_append_only_with_allowed_role",
    "loose_tool_call_message_matches",
    "resolve_session_message_matcher",
    "role_content_only_message_matches",
    "strict_message_matches",
]
