# [MODULE DOC]
# Typed exceptions for the session server. All derive from one base that carries a class-level
# HTTP status code, which the server uses to turn a raised error into an HTTP response: not found
# 404, message validation 400, truncated generation 409, tokenization 500, upstream invalid
# response 502, upstream aborted generation 503; the base default is 500.
"""Error types for the session module.

Hierarchy
---------
SessionError (base)
├── SessionNotFoundError       → 404  session does not exist
├── MessageValidationError     → 400  messages structure/content invalid
├── TruncatedGenerationError   → 409  extending a length-truncated generation (v2)
├── TokenizationError          → 500  TITO tokenizer / prefix mismatch
├── UpstreamResponseError      → 502  SGLang response invalid or unexpected
└── UpstreamGenerationAbortedError → 503  SGLang aborted the generation
"""


class SessionError(Exception):
    # [CLASS DOC] Base session error; the status code attribute (default 500) is the HTTP status
    # to report. Subclasses only override the status code.
    """Base class for all session-related errors."""

    status_code: int = 500


class SessionNotFoundError(SessionError):
    # [CLASS DOC] Status 404: the session id is unknown.
    """Raised when the requested session ID does not exist."""

    status_code: int = 404


class MessageValidationError(SessionError):
    # [CLASS DOC] Status 400: the request's message list is structurally invalid.
    """Raised when request messages fail structural validation.

    Examples: user message after assistant, messages not append-only,
    rollback failed (no assistant checkpoint in matched prefix).
    """

    status_code: int = 400


class TruncatedGenerationError(SessionError):
    # [CLASS DOC] Status 409: the request would extend a generation that stopped because of the
    # length limit.
    """Raised when a request extends a generation that ended with
    finish_reason='length'; only the v2 tree server raises it (v1 never
    branches)."""

    status_code: int = 409


class TokenizationError(SessionError):
    # [CLASS DOC] Status 500: token-in token-out invariant violated (e.g. stored prefix tokens do
    # not match the newly tokenized prefix).
    """Raised when TITO tokenization invariants are violated.

    Examples: pretokenized prefix mismatch between stored and new token IDs.
    """

    status_code: int = 500


class UpstreamResponseError(SessionError):
    # [CLASS DOC] Status 502: the SGLang backend answered with something invalid or unexpected.
    """Raised when the upstream SGLang response is invalid or unexpected.

    Examples: missing meta_info, assistant content is None,
    output_token_logprobs length mismatch.
    """

    status_code: int = 502


class UpstreamGenerationAbortedError(SessionError):
    # [CLASS DOC] Status 503: the backend aborted the generation although the HTTP call succeeded.
    """Raised when upstream aborts a generation without an HTTP error."""

    status_code: int = 503
