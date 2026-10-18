# [MODULE DOC] Command-line tool (typer) that generates the JSON schema files for the Helm
# charts from the Python values models. Three outputs, relative to the repo root: the shared
# infra schema (charts/shared-infra.schema.json), and values.schema.json for the miles-run and
# miles-workbench charts. Schemas are draft-07: each model's schema is produced with the
# model library, definitions moved to the draft-07 location, titles and defaults dropped
# recursively, and optional-or-null alternatives collapsed. Chart roots forbid unknown keys but
# explicitly allow the helm-injected global section and the library-chart section. With the check
# flag, nothing is written and the process exits with code 1 if any file differs. Importing
# prepends the repo root to the module search path (side effect).
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from miles.utils.external_utils.command_utils.helm_backend.launcher.values.helm_values_types import (  # noqa: E402
    InfraValues,
    MilesRunChartValues,
    MilesWorkbenchChartValues,
    ValuesModel,
)

DRAFT_07 = "http://json-schema.org/draft-07/schema#"

_SHARED_INFRA_PATH = Path("charts/shared-infra.schema.json")
_SHARED_INFRA_DESCRIPTION = (
    "Cluster-shaped values that every Miles chart accepts under the same paths, so one user-maintained "
    "values.yaml drives miles-workbench and miles-run alike. Helm does not resolve cross-file $ref, so each "
    "chart inlines these definitions verbatim into its own values.schema.json; "
    "tools/kubernetes/generate_chart_schema.py writes all three from the same Python types."
)

# Helm hands every chart these two sections whether or not the chart declares them, so a root that
# forbids unknown keys has to name them itself.
_HELM_SECTIONS: dict[str, Any] = {
    "global": {
        "type": "object",
        "description": (
            "Helm injects the umbrella chart's global values into every subchart, so a strict root has to "
            "accept them."
        ),
    },
    "miles-common": {
        "type": "object",
        "additionalProperties": False,
        "properties": {"global": {"type": "object"}},
        "description": "Helm reserves a values section per dependency; the library chart itself consumes none.",
    },
}

app = typer.Typer(add_completion=False)


@app.command()
def main(
    check: Annotated[bool, typer.Option(help="Fail instead of writing when a file is out of date")] = False,
) -> None:
    # [FUNCTION DOC] Command entry. Renders all schemas and, for each, writes it or (check mode)
    # reports staleness (every file is processed, none short-circuited, since the list is built
    # eagerly). Raises a typer exit with code 1 if any file was out of date in check mode.
    # In write mode the files are updated and the exit code is zero.
    stale = [path for path, content in generated_schemas().items() if not _write(path, content, check=check)]
    if stale:
        raise typer.Exit(code=1)


def generated_schemas() -> dict[Path, str]:
    # [FUNCTION DOC] Returns a mapping from repo-relative path to rendered schema text for the
    # shared infra schema, the miles-run chart (title "miles-run values", required sections
    # infra and run) and the miles-workbench chart (title "miles-workbench values", required
    # section infra).
    return {
        _SHARED_INFRA_PATH: _rendered(_shared_infra_schema()),
        Path("charts/miles-run/values.schema.json"): _rendered(
            _chart_schema(MilesRunChartValues, title="miles-run values", required=["infra", "run"])
        ),
        Path("charts/miles-workbench/values.schema.json"): _rendered(
            _chart_schema(MilesWorkbenchChartValues, title="miles-workbench values", required=["infra"])
        ),
    }


def _write(path: Path, content: str, *, check: bool) -> bool:
    # [FUNCTION DOC] HELPER: Returns true if the file already has exactly this content. Else, in
    # check mode prints a stale message and returns false; otherwise overwrites the file (the
    # directory must exist), prints a wrote message and returns true.
    absolute = REPO_ROOT / path
    if absolute.exists() and absolute.read_text() == content:
        return True
    if check:
        print(f"{path} is out of date; rerun tools/kubernetes/generate_chart_schema.py", flush=True)
        return False
    absolute.write_text(content)
    print(f"Wrote {path}", flush=True)
    return True


def _shared_infra_schema() -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Builds the draft-07 schema for the infra values model, moves its
    # definitions to the top level, and wraps it as the "infra" property of an object schema with
    # a title and the shared description. Definitions are emitted only if non-empty.
    schema = _draft_07(InfraValues)
    definitions = schema.pop("definitions", None)
    return {
        "$schema": DRAFT_07,
        "title": "Miles shared infra values",
        "description": _SHARED_INFRA_DESCRIPTION,
        "type": "object",
        "properties": {"infra": schema},
        **({"definitions": definitions} if definitions else {}),
    }


def _chart_schema(values: type[ValuesModel], *, title: str, required: list[str]) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Builds a chart root schema: object, additional properties forbidden,
    # given required keys, properties being the Helm-provided sections merged with the model's
    # properties (the model's win on name clash), plus definitions if the model has any.
    schema = _draft_07(values)
    return {
        "$schema": DRAFT_07,
        "title": title,
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": {**_HELM_SECTIONS, **schema["properties"]},
        **({"definitions": schema["definitions"]} if "definitions" in schema else {}),
    }


def _draft_07(model: type[ValuesModel]) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Gets the model's JSON schema using field aliases and references of
    # the form "#/definitions/<model>", detaches the definitions section (default-location
    # key), simplifies the main schema and each definition, and re-attaches definitions under the
    # draft-07 key when any exist.
    schema = model.model_json_schema(by_alias=True, ref_template="#/definitions/{model}")
    definitions = schema.pop("$defs", None)
    simplified = _simplify(schema)
    if definitions:
        simplified["definitions"] = {name: _simplify(entry) for name, entry in definitions.items()}
    return simplified


_DROPPED_KEYS = frozenset({"title", "default"})


def _simplify(schema: Any) -> Any:
    # [FUNCTION DOC] HELPER: Recursive cleanup. Lists are simplified element-wise; non-mappings
    # returned as is; for mappings, first collapses an optional-or-null alternative list, then
    # drops title and default keys and simplifies the values recursively (including property
    # maps, where a property literally named title or default would also be dropped).
    # KNOWN ISSUE: the key filter applies at every mapping level, so a property named "title" or
    # "default" in a properties map is removed from the schema.
    if isinstance(schema, list):
        return [_simplify(entry) for entry in schema]
    if not isinstance(schema, dict):
        return schema

    schema = _without_null_branch(schema)
    return {key: _simplify(value) for key, value in schema.items() if key not in _DROPPED_KEYS}


def _without_null_branch(schema: dict[str, Any]) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: If the schema has an any-of list with a null-type alternative and
    # exactly one other alternative, returns the schema without the any-of key merged with that
    # remaining alternative (alternative's keys win). Otherwise returns the schema unchanged
    # (no list, no null alternative, or several remaining alternatives).
    branches = schema.get("anyOf")
    if not isinstance(branches, list):
        return schema

    kept = [branch for branch in branches if branch != {"type": "null"}]
    if len(kept) != 1 or len(kept) == len(branches):
        return schema
    return {**{key: value for key, value in schema.items() if key != "anyOf"}, **kept[0]}


def _rendered(schema: dict[str, Any]) -> str:
    # [FUNCTION DOC] HELPER: Serializes the schema as JSON indented by 2 spaces with a trailing
    # newline (key order preserved).
    return json.dumps(schema, indent=2) + "\n"


if __name__ == "__main__":
    app()
