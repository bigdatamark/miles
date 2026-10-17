# [MODULE DOC] Command line script run by the release workflow when cutting a release branch.
# It writes a release lock JSON file (named release-lock.json, in the current working
# directory) that pins the release branch name, the exact commits of the sglang and Megatron
# dependency repositories, the CI image tag, and audit-only fingerprints of the prebuilt wheels
# for the x86_64 and aarch64 CUDA 13.0 builds. Takes six positional arguments in the order
# given in the usage text below; there is no argument validation beyond the dependency SHAs.
"""Write release-lock.json for a release branch cut.

Usage: write_release_lock.py <branch> <sglang_sha> <megatron_sha> <ci_image_tag> <fp_cu130_x86> <fp_cu130_arm64>
"""

import json
import sys


def main() -> int:
    # [FUNCTION DOC] Entry point. Unpacks exactly six command line arguments (branch, sglang
    # commit, Megatron commit, CI image tag, x86_64 wheels fingerprint, aarch64 wheels
    # fingerprint); a different count raises a value error from unpacking. Asserts that both
    # commit strings are 40 characters long (a full hex SHA), failing with an assertion error
    # saying the dependency SHAs could not be resolved. Writes release-lock.json in the current
    # directory, indented by two spaces, containing: release_branch, fixed sglang repo
    # (sgl-project/sglang) and branch (sglang-miles), the sglang commit, fixed Megatron repo
    # (radixark/Megatron-LM) and branch (miles-main), the Megatron commit, ci_image_tag, and the
    # two wheel fingerprint fields. Always returns 0.
    # KNOWN ISSUE: the output file handle is opened inline and never explicitly closed or
    # flushed; it relies on interpreter finalization to flush the data.
    branch, sglang, megatron, image, fp_x86, fp_arm = sys.argv[1:]
    assert len(sglang) == 40 and len(megatron) == 40, "failed to resolve dependency SHAs"
    json.dump(
        {
            "release_branch": branch,
            "sglang_repo": "sgl-project/sglang",
            "sglang_branch": "sglang-miles",
            "sglang_commit": sglang,
            "megatron_repo": "radixark/Megatron-LM",
            "megatron_branch": "miles-main",
            "megatron_commit": megatron,
            "ci_image_tag": image,
            # Audit only: wheels rolling tags can have assets replaced in place,
            # so they cannot be pinned by name (see docker-build.yml).
            "wheels_fingerprint_cu130_x86_64": fp_x86,
            "wheels_fingerprint_cu130_aarch64": fp_arm,
        },
        open("release-lock.json", "w"),
        indent=2,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
