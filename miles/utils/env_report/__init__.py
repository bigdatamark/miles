# [PACKAGE DOC] Environment reporting for audit: collects facts about the running process (host,
# command line, parsed arguments, environment variables with secrets redacted, an optional
# launcher-written report), installed Python packages (with editable installs and their git
# state), and logs them as audit events. Modules: collector (gathers a report), git_state (git
# commit/diff/untracked-file fingerprint), launcher_report (reads the launcher's json report),
# redaction (hashing of secrets), reporter (background thread that logs reports periodically).
