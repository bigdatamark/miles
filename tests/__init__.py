# [PACKAGE DOC] Root package of the miles test suite. It only makes the tests directory importable
# so that test modules and CI helpers can use absolute imports such as the tests.ci modules; it
# defines no names. Subpackages hold fast CPU tests, GPU tests, end-to-end tests, manual tests and
# the CI tooling.

