#!/usr/bin/env python3
# [MODULE DOC]
# CI job script that runs arbitrary SQL against a Neon PostgreSQL database and writes a JSON
# result file for a later workflow step to encrypt and return. Everything comes from environment
# variables: REQUEST_ID (must be a canonical lowercase UUID), MAX_RESULT_BYTES (positive integer
# cap on the total encoded size of returned rows), SQL_GZIP_BASE64 (the SQL text, gzip compressed
# then base64 encoded), NEON_DATABASE_URL (connection string, never printed), RUN_ID, ACTOR,
# REASON and optional RESULT_PATH (default result.json in the working directory). The SQL is
# executed unmodified in autocommit mode, so it may contain multiple statements. The result
# document records the request metadata, a hash of the SQL, one result set per statement and a
# truncation flag. The psycopg driver is imported lazily and passed into the execution function
# so the logic can be tested with a fake driver.

import base64
import gzip
import hashlib
import json
import math
import os
import uuid
from pathlib import Path

SCHEMA_VERSION = 1


def decode_sql(sql_gzip_base64):
    # [FUNCTION DOC] Decodes the SQL transport format: strict base64 decoding (invalid characters
    # raise an error), gzip decompression, then UTF-8 decoding. Returns the SQL text. Errors from
    # the base64, gzip or unicode layers propagate.
    compressed = base64.b64decode(sql_gzip_base64, validate=True)
    return gzip.decompress(compressed).decode("utf-8")


def encode_value(value):
    # [FUNCTION DOC] Converts a database value into something strictly JSON serialisable. None,
    # booleans, integers and strings pass through; finite floats pass through, while non-finite
    # floats become a mapping with type float and the textual value; bytes become a mapping with
    # type bytes and base64 text; lists and tuples become lists with elements converted
    # recursively; dictionaries get string keys and converted values; any other object (dates,
    # decimals, UUIDs and so on) becomes a mapping with its type name and its string form.
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"type": "float", "value": str(value)}
    if isinstance(value, bytes):
        return {"type": "bytes", "base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, list):
        return [encode_value(item) for item in value]
    if isinstance(value, tuple):
        return [encode_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): encode_value(item) for key, item in value.items()}
    return {"type": type(value).__name__, "value": str(value)}


def _base_result(*, request_id, run_id, actor, reason, sql_text):
    # [FUNCTION DOC] HELPER: Creates the initial result document: schema version, request id, run id
    # as text, actor, reason, the hex SHA-256 of the UTF-8 SQL text, status ok, an empty list of
    # results, truncated false and no error.
    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "run_id": str(run_id),
        "actor": actor,
        "reason": reason,
        "sql_sha256": hashlib.sha256(sql_text.encode("utf-8")).hexdigest(),
        "status": "ok",
        "results": [],
        "truncated": False,
        "error": None,
    }


def execute_request(
    driver,
    *,
    dsn,
    sql_text,
    request_id,
    run_id,
    actor,
    reason,
    max_result_bytes,
):
    # [FUNCTION DOC] Runs the SQL and returns the result document (a mapping). Keyword-only
    # arguments besides the driver module: connection string, SQL text, request id, run id, actor,
    # reason and the maximum number of bytes of encoded row data to return. Connects with
    # autocommit and prepared statements disabled, executes the text without a prepared statement,
    # and iterates every result set the statements produce. Each result set records the command
    # status message, column names (empty if the statement returns no rows) and the encoded rows.
    # Rows are encoded with the value encoder; each row's size is its compact UTF-8 JSON length.
    # Once adding a row would exceed the byte limit, that row and all later rows (of this and later
    # sets) are dropped and the truncated flag is set; result sets themselves are still recorded.
    # A database driver error sets status error and an error mapping with the exception type name,
    # SQL state (if any) and message; results gathered before the error are kept. Other exceptions
    # propagate. Side effect: database access, which may modify data since any SQL is allowed.
    result = _base_result(
        request_id=request_id,
        run_id=run_id,
        actor=actor,
        reason=reason,
        sql_text=sql_text,
    )
    returned_bytes = 0
    try:
        with driver.connect(dsn, autocommit=True, prepare_threshold=None) as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql_text, prepare=False)
                for current in cursor.results():
                    columns = (
                        [column.name for column in current.description] if current.description is not None else []
                    )
                    result_set = {
                        "command_status": current.statusmessage,
                        "columns": columns,
                        "rows": [],
                    }
                    if current.description is not None:
                        for row in current:
                            if result["truncated"]:
                                continue
                            encoded_row = [encode_value(value) for value in row]
                            row_bytes = len(
                                json.dumps(
                                    encoded_row,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                    allow_nan=False,
                                ).encode("utf-8")
                            )
                            if returned_bytes + row_bytes > max_result_bytes:
                                result["truncated"] = True
                                continue
                            result_set["rows"].append(encoded_row)
                            returned_bytes += row_bytes
                    result["results"].append(result_set)
    except driver.Error as error:
        result["status"] = "error"
        result["error"] = {
            "type": type(error).__name__,
            "sqlstate": getattr(error, "sqlstate", None),
            "message": str(error),
        }
    return result


def main():
    # [FUNCTION DOC] Entry point; returns the process exit status (zero if the result status is ok,
    # else one). Reads and validates the environment (request id must round-trip through the UUID
    # parser unchanged, otherwise the process exits with a message; the byte limit must parse as an
    # integer and be positive), decodes the SQL, executes it with the real psycopg driver, writes the
    # result document as indented UTF-8 JSON (non-finite numbers disallowed) with a trailing newline
    # to the result path, and prints one summary line with the request id, status, number of
    # result sets and truncation flag (the SQL and rows are never printed). Missing variables raise
    # key errors.
    import psycopg

    request_id = os.environ["REQUEST_ID"]
    if str(uuid.UUID(request_id)) != request_id:
        raise SystemExit("REQUEST_ID must be a canonical UUID")
    max_result_bytes = int(os.environ["MAX_RESULT_BYTES"])
    if max_result_bytes <= 0:
        raise SystemExit("MAX_RESULT_BYTES must be positive")

    sql_text = decode_sql(os.environ["SQL_GZIP_BASE64"])
    result = execute_request(
        psycopg,
        dsn=os.environ["NEON_DATABASE_URL"],
        sql_text=sql_text,
        request_id=request_id,
        run_id=os.environ["RUN_ID"],
        actor=os.environ["ACTOR"],
        reason=os.environ["REASON"],
        max_result_bytes=max_result_bytes,
    )
    result_path = Path(os.environ.get("RESULT_PATH", "result.json"))
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Neon request {request_id} finished with status={result['status']} "
        f"result_sets={len(result['results'])} truncated={result['truncated']}"
    )
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
