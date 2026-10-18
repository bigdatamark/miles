# [MODULE DOC] Pytest configuration for the loss tests: registers two command-line switches used
# by the snapshot tests, one to generate snapshots from current code and one to compare current
# code against saved snapshots.
def pytest_addoption(parser):
    # [FUNCTION DOC] Pytest hook; adds a boolean switch to generate snapshots and a boolean switch to
    # compare against saved snapshots. Both default to off; they are not mutually exclusive here.
    parser.addoption("--snapshot", action="store_true", help="Generate snapshots from current code")
    parser.addoption("--compare", action="store_true", help="Compare current code against saved snapshots")
