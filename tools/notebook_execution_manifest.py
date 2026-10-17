#!/usr/bin/env python3
"""Extract auditable execution evidence from a saved Jupyter notebook.

This script reads notebook metadata, execution counters and stored outputs. It
does not import the notebook's dependencies or execute any notebook cell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def normalized_preview(value: str, limit: int = 240) -> str:
    compact = " ".join(value.split())
    return compact if len(compact) <= limit else f"{compact[: limit - 3]}..."


def source_text(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def output_preview(output: dict[str, Any]) -> str | None:
    if output.get("output_type") == "error":
        return normalized_preview(
            f"{output.get('ename', 'Error')}: {output.get('evalue', '')}"
        )

    text = output.get("text")
    if text:
        return normalized_preview("".join(text) if isinstance(text, list) else str(text))

    plain = output.get("data", {}).get("text/plain")
    if plain:
        return normalized_preview("".join(plain) if isinstance(plain, list) else str(plain))

    return None


def build_manifest(notebook_path: Path) -> dict[str, Any]:
    raw = notebook_path.read_bytes()
    notebook = json.loads(raw)
    metadata = notebook.get("metadata", {})

    cells: list[dict[str, Any]] = []
    retained_counts: list[int] = []
    without_count: list[int] = []

    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue

        source = source_text(cell)
        execution_count = cell.get("execution_count")
        if isinstance(execution_count, int):
            retained_counts.append(execution_count)
        else:
            without_count.append(index)

        outputs = cell.get("outputs", [])
        previews = [preview for item in outputs if (preview := output_preview(item))]
        first_line = next((line.strip() for line in source.splitlines() if line.strip()), "")

        cells.append(
            {
                "notebook_index": index,
                "execution_count": execution_count,
                "source_first_line": first_line,
                "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "output_types": [item.get("output_type") for item in outputs],
                "stored_output_previews": previews,
            }
        )

    maximum = max(retained_counts, default=0)
    present = set(retained_counts)
    missing_counts = [count for count in range(1, maximum + 1) if count not in present]

    return {
        "schema_version": 1,
        "evidence_source": "saved notebook metadata, execution_count fields and stored outputs",
        "notebook": notebook_path.name,
        "notebook_sha256": hashlib.sha256(raw).hexdigest(),
        "environment_metadata": {
            "accelerator": metadata.get("accelerator"),
            "colab_gpu_type": metadata.get("colab", {}).get("gpuType"),
            "kaggle_accelerator": metadata.get("kaggle", {}).get("accelerator"),
            "kaggle_docker_image_version_id": metadata.get("kaggle", {}).get(
                "dockerImageVersionId"
            ),
            "kaggle_gpu_enabled": metadata.get("kaggle", {}).get("isGpuEnabled"),
            "python_version": metadata.get("language_info", {}).get("version"),
        },
        "retained_execution_counts": sorted(retained_counts),
        "gaps_in_retained_counts": missing_counts,
        "code_cells_without_retained_count": without_count,
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = build_manifest(args.notebook)
    rendered = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
