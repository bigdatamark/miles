# [MODULE DOC]
# Offline tests of the "Neon access" tooling: the workflow-side job script (which runs arbitrary
# SQL against a Postgres driver and builds a result record), the client script that packs SQL
# for transport and decrypts/validates the encrypted result, and static checks of the workflow
# definition file. Both scripts are loaded by file path from the repository root (three levels
# above this file). The database driver, connection, cursor and result sets are replaced by small
# in-memory fakes, so no network, database or secret is used; only the CMS round-trip test needs
# the openssl command line tool. Registers itself as a CPU test in the stage-a-cpu suite.
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

ROOT = Path(__file__).parents[3]


def load_module(name, path):
    # [FUNCTION DOC] HELPER: Load and execute a Python source file as a module with the given name
    # from the given path, without registering it in the module table. Returns the module object.
    # Side effect: runs the file's top-level code.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


JOB = load_module("neon_access_job", ROOT / ".github/workflows/scripts/neon_access_job.py")
CLIENT = load_module(
    "run_neon_workflow",
    ROOT / ".claude/skills/neon-access/scripts/run_neon_workflow.py",
)


class FakeResult:
    # [CLASS DOC] HELPER: Fake of one statement result from the driver. Holds the command status
    # message, a description (list of column objects with a name, or nothing when there are no
    # columns) and the rows; iterating yields the rows.
    def __init__(self, statusmessage, columns=(), rows=()):
        # [FUNCTION DOC] HELPER: Store the status message, build the description from the column
        # names (nothing if no columns were given) and keep the rows.
        self.statusmessage = statusmessage
        self.description = [SimpleNamespace(name=column) for column in columns] if columns else None
        self.rows = rows

    def __iter__(self):
        # [FUNCTION DOC] HELPER: Iterate over the stored rows.
        return iter(self.rows)


class FakeCursor:
    # [CLASS DOC] HELPER: Fake database cursor usable as a context manager. Records the executed
    # SQL text and prepare flag, optionally raises a configured error on execute, and hands out an
    # iterator over the configured result sets.
    def __init__(self, result_sets, error=None):
        # [FUNCTION DOC] HELPER: Keep the list of result sets and the optional error to raise;
        # nothing has been executed yet.
        self.result_sets = result_sets
        self.error = error
        self.executed = None

    def __enter__(self):
        # [FUNCTION DOC] HELPER: Context manager entry; returns the cursor itself.
        return self

    def __exit__(self, *_args):
        # [FUNCTION DOC] HELPER: Context manager exit; never suppresses exceptions.
        return False

    def execute(self, sql_text, *, prepare):
        # [FUNCTION DOC] HELPER: Record the SQL text and the prepare flag, then raise the
        # configured error if there is one.
        self.executed = (sql_text, prepare)
        if self.error is not None:
            raise self.error

    def results(self):
        # [FUNCTION DOC] HELPER: Return an iterator over the configured result sets, one per
        # statement of the executed SQL.
        return iter(self.result_sets)


class FakeConnection:
    # [CLASS DOC] HELPER: Fake database connection usable as a context manager that always hands
    # out the one configured cursor.
    def __init__(self, cursor):
        # [FUNCTION DOC] HELPER: Keep the cursor to hand out.
        self.fake_cursor = cursor

    def __enter__(self):
        # [FUNCTION DOC] HELPER: Context manager entry; returns the connection itself.
        return self

    def __exit__(self, *_args):
        # [FUNCTION DOC] HELPER: Context manager exit; never suppresses exceptions.
        return False

    def cursor(self):
        # [FUNCTION DOC] HELPER: Return the configured fake cursor.
        return self.fake_cursor


class FakeDriver:
    # [CLASS DOC] HELPER: Fake database driver module. Exposes a nested database error type and a
    # connect function that records its arguments and returns a fake connection.
    class Error(Exception):
        # [CLASS DOC] HELPER: Stand-in for the driver's base database error type.
        pass

    def __init__(self, cursor):
        # [FUNCTION DOC] HELPER: Build the fake connection around the cursor and remember that no
        # connect call has happened yet.
        self.fake_connection = FakeConnection(cursor)
        self.connect_call = None

    def connect(self, dsn, **kwargs):
        # [FUNCTION DOC] HELPER: Record the connection string and keyword options, and return the
        # fake connection.
        self.connect_call = (dsn, kwargs)
        return self.fake_connection


def execute(driver, sql_text, *, max_result_bytes=4096):
    # [FUNCTION DOC] HELPER: Run the job script's request executor against the given fake driver
    # with fixed placeholder connection string, request id, run id "123", actor "writer" and
    # reason. The result size limit defaults to 4096 bytes. Returns the executor's result record.
    return JOB.execute_request(
        driver,
        dsn="postgresql://not-used",
        sql_text=sql_text,
        request_id="00000000-0000-0000-0000-000000000001",
        run_id="123",
        actor="writer",
        reason="offline test",
        max_result_bytes=max_result_bytes,
    )


def test_arbitrary_multi_statement_sql_is_passed_once_and_unchanged():
    # [FUNCTION DOC] Runs a multi-statement SQL text (table creation, insert with a semicolon in a
    # string, an anonymous block with a semicolon in its body, a select) through the executor with
    # four fake result sets. Asserts the cursor received the whole text once, unsplit, with
    # prepare off; the connect call used the given connection string with autocommit on and the
    # prepare threshold disabled; the result lists the four command statuses in order; the last
    # result's rows are the expected row; and the recorded SQL hash is the SHA-256 of the text.
    sql_text = """CREATE TEMP TABLE t(v text);
INSERT INTO t VALUES ('do not split; this');
DO $$ BEGIN RAISE NOTICE 'inside;body'; END $$;
SELECT v FROM t;
"""
    cursor = FakeCursor(
        [
            FakeResult("CREATE TABLE"),
            FakeResult("INSERT 0 1"),
            FakeResult("DO"),
            FakeResult("SELECT 1", ["v"], [("do not split; this",)]),
        ]
    )
    driver = FakeDriver(cursor)

    result = execute(driver, sql_text)

    assert cursor.executed == (sql_text, False)
    assert driver.connect_call == (
        "postgresql://not-used",
        {"autocommit": True, "prepare_threshold": None},
    )
    assert [item["command_status"] for item in result["results"]] == [
        "CREATE TABLE",
        "INSERT 0 1",
        "DO",
        "SELECT 1",
    ]
    assert result["results"][-1]["rows"] == [["do not split; this"]]
    assert result["sql_sha256"] == hashlib.sha256(sql_text.encode()).hexdigest()


def test_result_encoding_and_limit_do_not_change_execution():
    # [FUNCTION DOC] Executes a query whose single result has three rows (a bytes value, a nested
    # mapping containing a tuple, a 100-character string) with a 70-byte result limit. Asserts the
    # SQL was still executed unchanged, bytes are encoded as a typed base64 object, tuples become
    # lists, rows beyond the size limit are dropped (the third row is cut), the result is flagged
    # truncated, and the status is still ok.
    cursor = FakeCursor(
        [
            FakeResult(
                "SELECT 3",
                ["payload"],
                [
                    (b"binary",),
                    ({"nested": (1, 2)},),
                    ("x" * 100,),
                ],
            )
        ]
    )
    driver = FakeDriver(cursor)

    result = execute(driver, "SELECT arbitrary_result", max_result_bytes=70)

    assert cursor.executed == ("SELECT arbitrary_result", False)
    assert result["results"][0]["rows"] == [
        [{"type": "bytes", "base64": "YmluYXJ5"}],
        [{"nested": [1, 2]}],
    ]
    assert result["truncated"] is True
    assert result["status"] == "ok"


def test_database_error_is_returned_in_result():
    # [FUNCTION DOC] Makes the cursor raise a driver error carrying an SQL state code. Asserts the
    # executor does not propagate it but returns status "error" with an error record holding the
    # error type name, the SQL state and the message.
    error = FakeDriver.Error("permission denied by PostgreSQL")
    error.sqlstate = "42501"
    driver = FakeDriver(FakeCursor([], error=error))

    result = execute(driver, "DROP TABLE protected")

    assert result["status"] == "error"
    assert result["error"] == {
        "type": "Error",
        "sqlstate": "42501",
        "message": "permission denied by PostgreSQL",
    }


def test_sql_transport_round_trip_preserves_utf8_bytes():
    # [FUNCTION DOC] Packs UTF-8 SQL bytes (non-ASCII character and a trailing newline) with the
    # client's packer, decodes with the job's decoder, and asserts the bytes round-trip exactly
    # and the returned digest is the SHA-256 of the original bytes.
    sql_bytes = "SELECT '雪';\n-- exact trailing newline\n".encode()

    packed, digest = CLIENT.pack_sql(sql_bytes)

    assert JOB.decode_sql(packed).encode() == sql_bytes
    assert digest == hashlib.sha256(sql_bytes).hexdigest()


def test_cms_round_trip_and_identity_validation(tmp_path):
    # [FUNCTION DOC] Creates a temporary recipient key and certificate with the client helper,
    # writes a result record to a file, encrypts it with the openssl command line tool (CMS,
    # AES-256, DER output; subprocess side effect, must succeed), then decrypts with the client
    # and validates the decrypted record against matching request id, run id, actor and SQL hash.
    # Asserts the decrypted record equals the original. All files live in the temporary directory.
    key_path, cert_path, _ = CLIENT._create_recipient(tmp_path)
    result = {
        "schema_version": 1,
        "request_id": "request",
        "run_id": "123",
        "actor": "writer",
        "sql_sha256": "digest",
        "status": "ok",
    }
    source_path = tmp_path / "result.json"
    encrypted_path = tmp_path / "result.json.cms"
    decrypted_path = tmp_path / "decrypted.json"
    source_path.write_text(json.dumps(result), encoding="utf-8")
    subprocess.run(
        [
            "openssl",
            "cms",
            "-encrypt",
            "-binary",
            "-aes256",
            "-in",
            str(source_path),
            "-outform",
            "DER",
            "-out",
            str(encrypted_path),
            str(cert_path),
        ],
        check=True,
    )

    CLIENT.decrypt_result(
        encrypted_path,
        decrypted_path,
        key_path,
        cert_path,
    )
    decrypted = json.loads(decrypted_path.read_text(encoding="utf-8"))
    CLIENT.validate_result(
        decrypted,
        request_id="request",
        run_id="123",
        actor="writer",
        sql_sha256="digest",
    )
    assert decrypted == result


def test_workflow_keeps_secret_and_sql_at_executor_boundary():
    # [FUNCTION DOC] Reads the Neon access workflow file as text and asserts static security
    # properties: the database secret is referenced exactly once; the SQL arrives as a compressed
    # base64 input; the job script is run by one step; the step summary is never written; the
    # collaborator permission of the triggering actor is checked and anything other than write
    # is refused; a default-branch-only guard message exists; the artifact retention is one day;
    # and the checkout and upload-artifact actions are pinned to exact commit hashes.
    workflow = (ROOT / ".github/workflows/neon-access.yml").read_text()

    assert workflow.count("secrets.NEON_DATABASE_URL") == 1
    assert "SQL_GZIP_BASE64: ${{ inputs.sql_gzip_base64 }}" in workflow
    assert "run: python3 .github/workflows/scripts/neon_access_job.py" in workflow
    assert "GITHUB_STEP_SUMMARY" not in workflow
    assert "collaborators/${actor}/permission" in workflow
    assert "github.triggering_actor" in workflow
    assert 'permission" != "write"' in workflow
    assert "Neon access must run from the default branch" in workflow
    assert "retention-days: 1" in workflow
    assert "actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683" in workflow
    assert "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02" in workflow
