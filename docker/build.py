#!/usr/bin/env python3
# doc-dev: docs/developer/ci/02-docker-build.md
# [MODULE DOC]
# Command line driver (typer based) that assembles and runs a docker buildx build for a Miles
# image. A table of variants maps a variant name to the target image repository, optional
# platform list, tag postfix and prefix, optional dockerfile override and variant build
# arguments (CUDA 13 multi- and single-architecture images, a Rubin arm64 image, and several ROCm
# images for different GPU generations). Tag modes: latest, dev (a moving tag plus a UTC
# timestamped tag) and custom (a user supplied tag). The build stamps the image with a label
# holding the content hash of the build inputs (from the image inputs module) so CI can skip
# unchanged rebuilds. Proxy settings from the environment are passed as build arguments. With the
# dry-run flag the commands are only printed. Needs the docker CLI with buildx and the typer
# package; sibling module image_inputs is imported by plain name so the script must run from its
# own directory context (run as a script).
"""Build and push Miles Docker images.

Usage:
    python docker/build.py --variant cu13 --image-tag dev --push          # multi-arch (amd64+arm64)
    python docker/build.py --variant cu13-x86 --image-tag dev --push      # single arch
    python docker/build.py --variant cu13 --image-tag dev --dry-run
"""

import os
import subprocess
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import image_inputs
import typer

CACHE_DIR = "/tmp/miles-docker-cache"
REPO_ROOT = Path(__file__).resolve().parent.parent

VARIANTS = {
    "cu13": {
        "image": "radixark/miles",
        "platforms": ["linux/amd64", "linux/arm64"],
        "tag_postfix": "",
        "build_args": {},
    },
    "cu13-x86": {
        "image": "radixark/miles",
        "platforms": ["linux/amd64"],
        "tag_postfix": "",
        "build_args": {},
    },
    "cu13-aarch64": {
        "image": "radixark/miles",
        "platforms": ["linux/arm64"],
        "tag_postfix": "",
        "build_args": {},
    },
    "rubin": {
        "image": "radixark/miles",
        "platforms": ["linux/arm64"],
        "tag_postfix": "-rubin",
        "dockerfile": "docker/Dockerfile.rubin",
        "build_args": {},
    },
    "rocm724-mi35x": {
        "image": "rocm/sgl-dev",
        "tag_postfix": "-rocm724-mi35x",
        "tag_prefix": "miles",
        "dockerfile": "docker/Dockerfile.rocm",
        "build_args": {
            "GPU_ARCH": "gfx950",
            "SGLANG_IMAGE_REPO": "lmsysorg/sglang",
            "SGLANG_IMAGE_TAG": "v0.5.21-rocm724-mi35x",
            "WHEELS_TAG_ROCM": "rocm724-gfx950-v0.5.20",
            "APPLY_ROCR_VMMFIX": "1",
            "TE_USE_WHEEL": "1",
        },
    },
    "rocm10-mi35x": {
        "image": "rocm/sgl-dev",
        "tag_postfix": "-rocm10-mi35x",
        "tag_prefix": "miles",
        "dockerfile": "docker/Dockerfile.rocm",
        "build_args": {
            "GPU_ARCH": "gfx950",
            "SGLANG_IMAGE_REPO": "lmsysorg/sglang",
            "SGLANG_IMAGE_TAG": "v0.5.21-rocm10-mi35x",
            "WHEELS_TAG_ROCM": "rocm10-gfx950-v0.5.18",
            "APEX_USE_PREBUILT": "1",
            "TE_USE_WHEEL": "1",
        },
    },
    "rocm10-mi30x": {
        "image": "rocm/sgl-dev",
        "tag_postfix": "-rocm10-mi30x",
        "tag_prefix": "miles",
        "dockerfile": "docker/Dockerfile.rocm",
        "build_args": {
            "GPU_ARCH": "gfx942",
            "SGLANG_IMAGE_REPO": "lmsysorg/sglang",
            "SGLANG_IMAGE_TAG": "v0.5.21-rocm10-mi30x",
            "WHEELS_TAG_ROCM": "rocm10-gfx942-v0.5.20",
            "APEX_USE_PREBUILT": "1",
            "TE_USE_WHEEL": "1",
            "AITER_PREBUILD_JIT": "1",
        },
    },
}


def run(cmd: list[str], dry_run: bool) -> None:
    # [FUNCTION DOC] HELPER: Prints the command (space-joined, prefixed by a plus) and, unless
    # dry-run is set, executes it as a subprocess, raising a called-process error on a non-zero
    # exit status.
    print(f"+ {' '.join(cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def build_and_push(
    variant: str,
    image_tag: str,
    dry_run: bool,
    dockerfile: str,
    push: bool = False,
    custom_tag: str = "",
    extra_build_args: list[str] | None = None,
    context: Path = REPO_ROOT,
    output: str = "",
) -> None:
    # [FUNCTION DOC] Builds the docker buildx command for a variant and runs it. Inputs: variant
    # name (key into the variant table; unknown raises a key error), tag mode, dry-run flag, default
    # dockerfile path (overridden by the variant's own dockerfile), push flag, custom tag, extra
    # KEY=VALUE build arguments, build context directory (default repository root) and an optional
    # buildx exporter specification. Tags: latest gives the image with the latest tag plus the
    # variant postfix; dev gives the prefix (default dev) with postfix and a second tag with a UTC
    # minute-resolution timestamp; custom requires a custom tag (else a typer bad-parameter error)
    # and gives that tag plus postfix; anything else is a bad-parameter error. Command layout:
    # buildx build, dockerfile path joined onto the context, platforms if the variant lists them,
    # push flag if set, output exporter if given, HTTP and HTTPS proxy build arguments if present in
    # the environment (lowercase name checked first), a no-proxy argument for localhost, variant
    # build arguments, caller build arguments last so they win (each must contain an equals sign,
    # asserted), the inputs-hash label computed from the context, one tag option per tag, and the
    # context path. Prints a banner and runs the command.
    extra_build_args = extra_build_args or []
    config = VARIANTS[variant]
    # A variant may pin its own Dockerfile (e.g. ROCm); otherwise use the CLI default.
    dockerfile = config.get("dockerfile", dockerfile)
    image = config["image"]
    postfix = config.get("tag_postfix", "")
    platforms = config.get("platforms")

    if image_tag == "latest":
        tags = [f"{image}:latest{postfix}"]
    elif image_tag == "dev":
        prefix = config.get("tag_prefix", "dev")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
        tags = [f"{image}:{prefix}{postfix}", f"{image}:{prefix}{postfix}-{timestamp}"]
    elif image_tag == "custom":
        if not custom_tag:
            raise typer.BadParameter("--custom-tag is required when --image-tag is custom")
        tags = [f"{image}:{custom_tag}{postfix}"]
    else:
        raise typer.BadParameter(f"Unknown image tag: {image_tag}")

    cmd = [
        "docker",
        "buildx",
        "build",
        "-f",
        str(context / dockerfile),
    ]

    if platforms:
        cmd += ["--platform", ",".join(platforms)]

    if push:
        cmd += ["--push"]
    if output:
        cmd += ["--output", output]

    # Proxy args (pass through if set in environment, check both cases)
    for arg_name in ["HTTP_PROXY", "HTTPS_PROXY"]:
        value = os.environ.get(arg_name.lower()) or os.environ.get(arg_name)
        if value:
            cmd += ["--build-arg", f"{arg_name}={value}"]

    cmd += ["--build-arg", "NO_PROXY=localhost,127.0.0.1"]

    # Variant-specific build args
    for key, value in config.get("build_args", {}).items():
        cmd += ["--build-arg", f"{key}={value}"]

    # Caller overrides (e.g. release builds pinning SGLANG_COMMIT / MILES_COMMIT
    # from release-lock.json) come last so they win over variant defaults.
    for spec in extra_build_args:
        assert "=" in spec, f"--build-arg expects KEY=VALUE, got {spec!r}"
        cmd += ["--build-arg", spec]

    # CI reads this back off the published tag to skip rebuilds whose inputs are unchanged.
    cmd += ["--label", f"{image_inputs.LABEL_KEY}={image_inputs.compute(root=context)}"]

    for tag in tags:
        cmd += ["-t", tag]

    # Context is repo root
    cmd += [str(context)]

    print(f"\n=== Building {' '.join(tags)} ===", flush=True)
    run(cmd, dry_run)


class Variant(str, Enum):
    # [CLASS DOC]
    # Enumeration of selectable build variants for the command line; each member's value is the key
    # of the variant table. The member list must be kept in sync with that table by hand.
    cu13 = "cu13"
    cu13_x86 = "cu13-x86"
    cu13_aarch64 = "cu13-aarch64"
    rubin = "rubin"
    rocm724_mi35x = "rocm724-mi35x"
    rocm10_mi35x = "rocm10-mi35x"
    rocm10_mi30x = "rocm10-mi30x"


class ImageTag(str, Enum):
    # [CLASS DOC]
    # Enumeration of tag modes accepted on the command line: latest, dev and custom.
    latest = "latest"
    dev = "dev"
    custom = "custom"


def main(
    variant: Variant = typer.Option(..., help="Build variant to use."),  # noqa: B008
    image_tag: ImageTag = typer.Option(..., help="Tag mode: latest, dev, or custom."),  # noqa: B008
    dockerfile: str = typer.Option("docker/Dockerfile", help="Path to the Dockerfile."),  # noqa: B008
    dry_run: bool = typer.Option(False, help="Print commands without executing them."),  # noqa: B008
    push: bool = typer.Option(False, help="Push images to registry after building."),  # noqa: B008
    custom_tag: str = typer.Option("", help="Custom tag name (required when --image-tag is custom)."),  # noqa: B008
    build_arg: list[str] = typer.Option([], help="Extra KEY=VALUE build-arg (repeatable)."),  # noqa: B008
    context: Path = typer.Option(REPO_ROOT, help="Repository build context, separate from this driver."),  # noqa: B008
    output: str = typer.Option("", help="Buildx exporter, e.g. type=oci,dest=/tmp/image,tar=false."),  # noqa: B008
) -> None:
    # [FUNCTION DOC] Typer command. Options: required variant and tag mode; dockerfile path
    # (default docker/Dockerfile, ignored when the variant pins its own); dry-run; push; custom tag;
    # repeatable extra build arguments; build context directory (default repository root; resolved
    # to an absolute path); and a buildx exporter specification. Passes everything to the builder
    # function using the enumeration values.
    build_and_push(
        variant.value,
        image_tag.value,
        dry_run,
        dockerfile,
        push=push,
        custom_tag=custom_tag,
        extra_build_args=build_arg,
        context=context.resolve(),
        output=output,
    )


if __name__ == "__main__":
    typer.run(main)
