# [MODULE DOC] Fingerprints the git state of a directory for the environment report: current
# commit, whether the work tree is dirty, the diff summary against HEAD, the list of untracked
# files (capped), and a SHA-256 over the full diff text plus per-file hashes of untracked files, so
# two reports can tell whether uncommitted contents are identical. Collection is retried once if
# HEAD changes while collecting. All git commands run with a 10 second timeout in the target
# directory. Limits: at most 1000 untracked files listed, files over 10 MiB are not content-hashed
# (their size is hashed instead).
import hashlib
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from miles.utils.audit_utils.event_logger.models import EnvReportGitRepoInfo

logger = logging.getLogger(__name__)
_HEAD_MOVE_ATTEMPTS = 2
_UNTRACKED_MAX_FILES = 1000
_UNTRACKED_MAX_FILE_BYTES = 10 * 1024 * 1024
_PATCH_SECTION_STARTS = (b"diff --git ", b"diff --cc ", b"diff --combined ", b"* Unmerged path ")


def collect_git_info(*, package_name: str, location: str) -> EnvReportGitRepoInfo | None:
    # [FUNCTION DOC] Returns none if the location is empty or not a directory. Otherwise tries up
    # to two times to collect a stable snapshot (a try is abandoned if HEAD moved during
    # collection); if both tries see HEAD move, logs a warning and returns none. Also returns none
    # (via the helper) when the directory is not a git repository or collection fails.
    if not location or not os.path.isdir(location):
        return None
    for _ in range(_HEAD_MOVE_ATTEMPTS):
        try:
            return _collect_stable_git_info(package_name=package_name, location=location)
        except _HeadMoved:
            continue
    logger.warning("Gave up collecting the git state of %s at %s, its HEAD keeps moving", package_name, location)
    return None


class _HeadMoved(Exception):
    # [CLASS DOC] HELPER: Signal raised when HEAD differs before and after collecting the diff and
    # untracked files; carries no data.
    pass


def _collect_stable_git_info(*, package_name: str, location: str) -> EnvReportGitRepoInfo | None:
    # [FUNCTION DOC] HELPER: Reads HEAD (a failing command, e.g. not a repository, gives none),
    # then the diff and the untracked files, then reads HEAD again and raises the head-moved signal
    # if the raw output changed. Returns the record: package name, location, commit hash, dirty
    # flag (non-empty diff summary or any untracked path), diff summary, uncommitted-content hash,
    # untracked paths, truncation flag and the list of untracked paths that were not content-hashed.
    # Any other exception is logged as a warning and gives none.
    try:
        commit_result = _run_git(args=["rev-parse", "HEAD"], location=location)
        if commit_result.returncode != 0:
            return None

        diff = _collect_diff(location=location)

        untracked = _collect_untracked(location=location)

        if _run_git(args=["rev-parse", "HEAD"], location=location).stdout != commit_result.stdout:
            raise _HeadMoved

        return EnvReportGitRepoInfo(
            package_name=package_name,
            location=location,
            commit=_decode(commit_result.stdout).strip(),
            dirty=bool(diff.stat) or bool(untracked.paths),
            diff_stat=diff.stat,
            uncommitted_hash=_hash_uncommitted(diff_patch=diff.patch, untracked=untracked),
            untracked_paths=untracked.paths,
            untracked_paths_truncated=untracked.paths_truncated,
            untracked_unhashed_paths=untracked.unhashed_paths,
        )
    except _HeadMoved:
        raise
    except Exception:
        logger.warning("Failed to collect git info for %s at %s", package_name, location, exc_info=True)
        return None


@dataclass(frozen=True)
class _GitDiff:
    # [CLASS DOC] HELPER: Diff summary text, and the raw patch bytes (none when the diff command
    # failed, empty when there are no patch sections).
    stat: str
    patch: bytes | None


def _collect_diff(*, location: str) -> _GitDiff:
    # [FUNCTION DOC] HELPER: Runs a diff of the work tree against HEAD with patch and summary. On
    # failure returns empty summary and no patch. If no patch section is found, the whole output
    # (stripped, decoded) is the summary and the patch is empty bytes; otherwise the summary is
    # the text before the first patch section and the patch is the rest.
    result = _run_git(args=["diff", "--patch-with-stat", "HEAD"], location=location)
    if result.returncode != 0:
        return _GitDiff(stat="", patch=None)

    if (patch_start := _find_patch_start(stdout=result.stdout)) is None:
        return _GitDiff(stat=_decode(result.stdout).strip(), patch=b"")

    return _GitDiff(stat=_decode(result.stdout[:patch_start]).strip(), patch=result.stdout[patch_start:])


def _find_patch_start(*, stdout: bytes) -> int | None:
    # [FUNCTION DOC] HELPER: Finds the byte offset of the earliest line beginning with one of the
    # patch section markers (normal, combined and unmerged-path headers); a newline is prepended
    # so a match at the very start is found. The found index equals the marker's offset in the
    # original output. Returns none when there is none.
    sentinelled = b"\n" + stdout
    starts = [found for start in _PATCH_SECTION_STARTS if (found := sentinelled.find(b"\n" + start)) >= 0]
    return min(starts) if starts else None


@dataclass(frozen=True)
class _UntrackedFiles:
    # [CLASS DOC] HELPER: Untracked-file listing: decoded paths (capped), a hash entry (bytes) per
    # path, whether the listing was truncated, the total number found, and the paths whose
    # content was not hashed.
    paths: list[str]
    hash_entries: list[bytes]
    paths_truncated: bool
    total_count: int
    unhashed_paths: list[str]


def _hash_uncommitted(*, diff_patch: bytes | None, untracked: _UntrackedFiles) -> str | None:
    # [FUNCTION DOC] HELPER: Returns none when the diff failed. Otherwise a SHA-256 hex digest over
    # a tagged, length-prefixed diff patch, then each untracked hash entry with a tag and
    # terminator, then (if truncated) a tag and the total file count.
    if diff_patch is None:
        return None

    digest = hashlib.sha256()
    digest.update(b"diff\0%d\0" % len(diff_patch))
    digest.update(diff_patch)
    for entry in untracked.hash_entries:
        digest.update(b"untracked\0" + entry + b"\0")
    if untracked.paths_truncated:
        digest.update(b"untracked-truncated\0%d\0" % untracked.total_count)

    return digest.hexdigest()


def _collect_untracked(*, location: str) -> _UntrackedFiles:
    # [FUNCTION DOC] HELPER: Resolves the repository root, lists untracked non-ignored files
    # (null-separated), sorts the raw paths, keeps the first 1000, and builds a hash entry for each
    # (recording the paths not content-hashed). A failing listing gives an empty result. Marks
    # truncation if more than 1000 were found and reports the total.
    root = _resolve_repo_root(location=location)

    result = _run_git(args=["ls-files", "--others", "--exclude-standard", "-z"], location=str(root))
    if result.returncode != 0:
        return _UntrackedFiles(paths=[], hash_entries=[], paths_truncated=False, total_count=0, unhashed_paths=[])

    all_paths = sorted(path for path in result.stdout.split(b"\0") if path)
    selected = all_paths[:_UNTRACKED_MAX_FILES]

    paths: list[str] = []
    hash_entries: list[bytes] = []
    unhashed_paths: list[str] = []
    for raw_path in selected:
        entry, hashed = _untracked_hash_entry(root=root, raw_path=raw_path)
        paths.append(_decode(raw_path))
        hash_entries.append(entry)
        if not hashed:
            unhashed_paths.append(_decode(raw_path))

    return _UntrackedFiles(
        paths=paths,
        hash_entries=hash_entries,
        paths_truncated=len(all_paths) > len(selected),
        total_count=len(all_paths),
        unhashed_paths=unhashed_paths,
    )


def _resolve_repo_root(*, location: str) -> Path:
    # [FUNCTION DOC] HELPER: The repository top-level directory reported by git, or the given
    # location itself if the command fails or prints nothing.
    result = _run_git(args=["rev-parse", "--show-toplevel"], location=location)
    if result.returncode != 0 or not (toplevel := _decode(result.stdout).strip()):
        return Path(location)
    return Path(toplevel)


def _untracked_hash_entry(*, root: Path, raw_path: bytes) -> tuple[bytes, bool]:
    # [FUNCTION DOC] HELPER: Returns (entry bytes, whether the content was hashed). Files over
    # 10 MiB: path, a zero byte and the size, not hashed. Otherwise path, a zero byte and the
    # SHA-256 of the whole file, hashed. On an operating system error (logged as a warning): path,
    # a zero byte and the word unreadable, not hashed. Reads the whole file into memory.
    full_path = root / os.fsdecode(raw_path)
    try:
        if (size := full_path.stat().st_size) > _UNTRACKED_MAX_FILE_BYTES:
            return b"%s\0size:%d" % (raw_path, size), False
        content_digest = hashlib.sha256(full_path.read_bytes()).hexdigest()
        return b"%s\0sha256:%s" % (raw_path, content_digest.encode()), True
    except OSError:
        logger.warning("Failed to hash the untracked file %s", full_path, exc_info=True)
        return b"%s\0unreadable" % raw_path, False


def _run_git(*, args: list[str], location: str) -> subprocess.CompletedProcess:
    # [FUNCTION DOC] HELPER: Runs git with the arguments in the given directory, capturing raw
    # byte output, 10 second timeout (a timeout or missing git executable raises). Does not check
    # the exit status.
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        timeout=10,
        cwd=location,
    )


def _decode(raw: bytes) -> str:
    # [FUNCTION DOC] HELPER: Decodes bytes to text, replacing undecodable bytes.
    return raw.decode(errors="replace")
