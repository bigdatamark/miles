#!/usr/bin/env python3
# [MODULE DOC] Command line client for the Neon database access workflow of the miles
# repository. Given a file of SQL text and a mandatory reason, it: checks the caller is
# authenticated with the GitHub command line tool and has write or admin permission on the
# repository; generates a throwaway one-day RSA certificate and private key in a private
# temporary directory; dispatches the neon-access GitHub Actions workflow on the default branch
# with the gzip+base64 packed SQL, the reason, the certificate and a maximum result size;
# watches the run; downloads the encrypted result artifact; decrypts it with the private key;
# validates that the result identity (request id, run id, actor, SQL hash) matches; tries to
# delete the artifact; and prints the result JSON. Requires the gh and openssl executables.
# Exit status is zero when the result status is ok and one when it is error.
import argparse
import base64
import gzip
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

REPOSITORY = "radixark/miles"
WORKFLOW = "neon-access.yml"
API_VERSION = "2026-03-10"
MAX_DISPATCH_CHARACTERS = 65_535
DEFAULT_MAX_RESULT_BYTES = 4 * 1024 * 1024


def _run(args, *, capture=True):
    # [FUNCTION DOC] HELPER: Runs an external command given as an argument list, raising the
    # called-process error on non-zero exit. Output is text. When capture is true standard
    # output is captured and available on the returned completed-process object; otherwise it
    # is inherited from the parent. Standard error is always inherited (never captured).
    return subprocess.run(
        args,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=None,
    )


def _gh_api(path, *, method="GET", fields=None):
    # [FUNCTION DOC] HELPER: Calls the GitHub REST API through the gh command line tool with
    # the given method, path and the pinned API version header. Each entry of the optional
    # fields mapping is passed as a string form field. Returns the parsed JSON of standard
    # output, or nothing when the output is blank (e.g. delete). Raises the called-process
    # error if gh fails; a JSON decode error if output is not JSON.
    args = [
        "gh",
        "api",
        "--method",
        method,
        "-H",
        f"X-GitHub-Api-Version: {API_VERSION}",
        path,
    ]
    for key, value in (fields or {}).items():
        args.extend(["-f", f"{key}={value}"])
    output = _run(args).stdout
    return json.loads(output) if output.strip() else None


def pack_sql(sql_bytes):
    # [FUNCTION DOC] Prepares SQL bytes for transport. Verifies they decode as UTF-8 (raises a
    # unicode decode error otherwise), compresses them with gzip using a zero timestamp for
    # deterministic output, and base64-encodes the result as ASCII text. Returns a pair: the
    # base64 text and the hex SHA-256 digest of the original uncompressed bytes.
    sql_bytes.decode("utf-8")
    compressed = gzip.compress(sql_bytes, mtime=0)
    return base64.b64encode(compressed).decode("ascii"), hashlib.sha256(sql_bytes).hexdigest()


def _create_recipient(temp_dir):
    # [FUNCTION DOC] HELPER: Creates a self-signed X.509 certificate (3072-bit RSA, no key
    # passphrase, subject name miles-neon-access, valid one day) by running openssl, writing
    # the private key and certificate as PEM files in the given directory. Restricts the key
    # file to owner read/write. Returns the key path, the certificate path, and the certificate
    # file contents base64-encoded as ASCII text. Raises the called-process error if openssl
    # fails.
    key_path = temp_dir / "recipient.key.pem"
    cert_path = temp_dir / "recipient.cert.pem"
    _run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:3072",
            "-nodes",
            "-subj",
            "/CN=miles-neon-access",
            "-days",
            "1",
            "-keyout",
            str(key_path),
            "-out",
            str(cert_path),
        ],
        capture=False,
    )
    key_path.chmod(0o600)
    cert_base64 = base64.b64encode(cert_path.read_bytes()).decode("ascii")
    return key_path, cert_path, cert_base64


def decrypt_result(encrypted_path, output_path, key_path, cert_path):
    # [FUNCTION DOC] Decrypts a binary DER-encoded CMS (encrypted message) file with openssl
    # using the recipient private key and certificate, writing plaintext to the output path.
    # Returns nothing; raises the called-process error if openssl fails.
    _run(
        [
            "openssl",
            "cms",
            "-decrypt",
            "-binary",
            "-inform",
            "DER",
            "-in",
            str(encrypted_path),
            "-inkey",
            str(key_path),
            "-recip",
            str(cert_path),
            "-out",
            str(output_path),
        ],
        capture=False,
    )


def validate_result(result, *, request_id, run_id, actor, sql_sha256):
    # [FUNCTION DOC] Checks that a decoded result mapping belongs to this request. It must
    # carry schema version 1 and equal the given request id, the run id (compared as text), the
    # actor login and the SQL hash. Any mismatch raises a runtime error naming the field. It
    # also raises a runtime error if the status is anything other than ok or error. Returns
    # nothing.
    expected = {
        "schema_version": 1,
        "request_id": request_id,
        "run_id": str(run_id),
        "actor": actor,
        "sql_sha256": sql_sha256,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise RuntimeError(f"result identity mismatch for {key}")
    if result.get("status") not in {"ok", "error"}:
        raise RuntimeError("result has an invalid status")


def _delete_artifact(run_id, artifact_name):
    # [FUNCTION DOC] HELPER: Lists the artifacts of the given workflow run, requires exactly
    # one with the given name (raises a runtime error otherwise, with the count found) and
    # deletes it through the API. Returns nothing. Only the first page of artifacts the API
    # returns is considered.
    response = _gh_api(f"repos/{REPOSITORY}/actions/runs/{run_id}/artifacts")
    matches = [item for item in response["artifacts"] if item["name"] == artifact_name]
    if len(matches) != 1:
        raise RuntimeError(f"expected one artifact named {artifact_name}, found {len(matches)}")
    _gh_api(
        f"repos/{REPOSITORY}/actions/artifacts/{matches[0]['id']}",
        method="DELETE",
    )


def _parse_args(argv):
    # [FUNCTION DOC] HELPER: Parses command line arguments: a required positional path to the
    # SQL file, a required reason string, and an optional maximum result size in bytes
    # (integer, default four mebibytes). Returns the parsed namespace; argparse exits on errors.
    parser = argparse.ArgumentParser(description="Run arbitrary SQL through the authorized Miles Neon workflow.")
    parser.add_argument("sql_file", type=Path)
    parser.add_argument("--reason", required=True)
    parser.add_argument(
        "--max-result-bytes",
        type=int,
        default=DEFAULT_MAX_RESULT_BYTES,
    )
    return parser.parse_args(argv)


def main(argv=None):
    # [FUNCTION DOC] Entry point implementing the whole flow described at module level.
    # Exits with a message if the maximum result size is not positive, if the caller's
    # permission is not write or admin, or if the serialized dispatch body would exceed the
    # 65535 character GitHub limit. Steps: read SQL bytes and pack them; make a random request
    # id and artifact name "neon-access-<id>"; check gh auth status; look up the actor login and
    # permission; fetch the repository default branch; inside a private (mode 0700) temporary
    # directory create the recipient key pair, build the workflow inputs (request id, packed
    # SQL, reason, certificate, max result bytes as text) and dispatch the workflow on the
    # default branch with each input as a form field. The dispatch response must be a mapping
    # with a workflow run id, else a runtime error is raised. Prints the run URL to standard
    # error, watches the run, downloads the artifact, decrypts result.json.cms, parses it,
    # validates identity, then tries to delete the artifact (failures only print a warning to
    # standard error). Prints the result as indented JSON to standard output. Returns 0 if the
    # result status is ok, otherwise 1. The temporary directory (with key) is removed on exit.
    # KNOWN ISSUE: the serialized-size check measures a body with nested inputs, but the real
    # dispatch sends them as separate form fields, so the limit check is only approximate.
    args = _parse_args(argv)
    if args.max_result_bytes <= 0:
        raise SystemExit("--max-result-bytes must be positive")

    sql_bytes = args.sql_file.read_bytes()
    sql_payload, sql_sha256 = pack_sql(sql_bytes)
    request_id = str(uuid.uuid4())
    artifact_name = f"neon-access-{request_id}"

    _run(["gh", "auth", "status"], capture=False)
    actor = _gh_api("user")["login"]
    permission = _gh_api(f"repos/{REPOSITORY}/collaborators/{actor}/permission")["permission"]
    if permission not in {"write", "admin"}:
        raise SystemExit(f"GitHub user {actor} has {permission!r}, not write/admin permission")

    repository = _gh_api(f"repos/{REPOSITORY}")
    with tempfile.TemporaryDirectory(prefix="miles-neon-access-") as temp_name:
        temp_dir = Path(temp_name)
        os.chmod(temp_dir, 0o700)
        key_path, cert_path, cert_base64 = _create_recipient(temp_dir)
        inputs = {
            "request_id": request_id,
            "sql_gzip_base64": sql_payload,
            "reason": args.reason,
            "recipient_cert_base64": cert_base64,
            "max_result_bytes": str(args.max_result_bytes),
        }
        dispatch_body = {
            "ref": repository["default_branch"],
            "inputs": inputs,
        }
        serialized = json.dumps(dispatch_body, separators=(",", ":"))
        if len(serialized) > MAX_DISPATCH_CHARACTERS:
            raise SystemExit(
                f"compressed request is {len(serialized)} characters; GitHub limit is " f"{MAX_DISPATCH_CHARACTERS}"
            )

        dispatch = _gh_api(
            f"repos/{REPOSITORY}/actions/workflows/{WORKFLOW}/dispatches",
            method="POST",
            fields={
                "ref": repository["default_branch"],
                **{f"inputs[{key}]": value for key, value in inputs.items()},
            },
        )
        if not isinstance(dispatch, dict) or "workflow_run_id" not in dispatch:
            raise RuntimeError("workflow dispatch did not return a run id; API version 2026-03-10 is required")
        run_id = str(dispatch["workflow_run_id"])
        run_url = dispatch.get("html_url") or dispatch.get("run_url")
        print(f"Workflow: {run_url}", file=sys.stderr)

        _run(["gh", "run", "watch", run_id, "--repo", REPOSITORY], capture=False)
        download_dir = temp_dir / "artifact"
        download_dir.mkdir()
        _run(
            [
                "gh",
                "run",
                "download",
                run_id,
                "--repo",
                REPOSITORY,
                "--name",
                artifact_name,
                "--dir",
                str(download_dir),
            ],
            capture=False,
        )
        encrypted_path = download_dir / "result.json.cms"
        result_path = temp_dir / "result.json"
        decrypt_result(encrypted_path, result_path, key_path, cert_path)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        validate_result(
            result,
            request_id=request_id,
            run_id=run_id,
            actor=actor,
            sql_sha256=sql_sha256,
        )

        try:
            _delete_artifact(run_id, artifact_name)
        except (subprocess.CalledProcessError, RuntimeError, TypeError, ValueError) as error:
            print(f"warning: failed to delete encrypted artifact: {error}", file=sys.stderr)

        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
