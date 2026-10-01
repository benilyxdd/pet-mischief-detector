"""Compute and plot per-class annotation counts."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Mapping

from src.constants.classes import CLASSES
from src.constants.paths import FIGURES_DIR
from src.utils.logging import get_logger
from src.utils.paths import ensure_dir

logger = get_logger(__name__)


def compute_class_counts(view, label_field: str = "ground_truth") -> Counter:
    counts: Counter = Counter()
    for sample in view.select_fields([label_field]).iter_samples(progress=True):
        dets = sample[label_field]
        if dets is None:
            continue
        for det in dets.detections:
            counts[det.label] += 1
    return counts


def plot_class_distribution(
    counts: Counter,
    output_path: Path | str = FIGURES_DIR / "class_distribution.png",
    title: str = "Class Distribution",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    labels = list(CLASSES)
    values = [counts.get(c, 0) for c in labels]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel("Number of instances")
    ax.tick_params(axis="x", rotation=45)
    for i, v in enumerate(values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info("Saved class distribution chart to %s", output_path)
    return output_path


def log_per_split_stats(
    splits: Mapping[str, object],
    label_field: str = "ground_truth",
) -> dict[str, Counter]:
    all_counts: dict[str, Counter] = {}
    for name, view in splits.items():
        counts = compute_class_counts(view, label_field=label_field)
        all_counts[name] = counts
        total = sum(counts.values())
        logger.info("Split '%s': %d samples, %d instances", name, len(view), total)
        for c in CLASSES:
            logger.info("  %-14s %d", c, counts.get(c, 0))
    return all_counts


def save_dataset_stats(
    splits: Mapping[str, object],
    per_split_counts: Mapping[str, Counter],
    output_dir: Path | str = FIGURES_DIR,
    label_field: str = "ground_truth",
) -> tuple[Path, Path]:
    """Persist per-class per-split annotation counts as JSON + Markdown.

    Two artefacts are produced under ``output_dir``:

    * ``dataset_stats.json`` -- machine-readable, with sample counts and the
      full ``{split: {class: count}}`` table plus totals.
    * ``dataset_stats.md`` -- a markdown table that the report can quote
      verbatim in section 3.7 ("Final dataset statistics").
    """
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    split_names = list(per_split_counts.keys())
    sample_counts = {name: len(splits[name]) for name in split_names}

    per_class: dict[str, dict[str, int]] = {}
    for cls in CLASSES:
        row = {name: int(per_split_counts[name].get(cls, 0)) for name in split_names}
        row["total"] = sum(row.values())
        per_class[cls] = row

    totals_per_split = {
        name: int(sum(per_split_counts[name].values())) for name in split_names
    }
    totals_per_split["total"] = sum(totals_per_split.values())

    payload = {
        "splits": split_names,
        "samples_per_split": sample_counts,
        "samples_total": sum(sample_counts.values()),
        "instances_per_split": totals_per_split,
        "per_class": per_class,
        "label_field": label_field,
    }

    json_path = output_dir / "dataset_stats.json"
    json_path.write_text(json.dumps(payload, indent=2))
    logger.info("Saved dataset stats JSON to %s", json_path)

    md_path = output_dir / "dataset_stats.md"
    _write_dataset_stats_md(payload, md_path)
    logger.info("Saved dataset stats markdown to %s", md_path)

    return json_path, md_path


def _write_dataset_stats_md(payload: dict, md_path: Path) -> None:
    split_names: list[str] = payload["splits"]
    sample_counts: dict[str, int] = payload["samples_per_split"]
    instances_per_split: dict[str, int] = payload["instances_per_split"]
    per_class: dict[str, dict[str, int]] = payload["per_class"]

    lines: list[str] = []
    lines.append("# Dataset statistics")
    lines.append("")
    lines.append(f"- **Total samples**: {payload['samples_total']}")
    lines.append(f"- **Total annotated instances**: {instances_per_split.get('total', 0)}")
    lines.append("")
    lines.append("## Per-split sample counts")
    lines.append("")
    header_splits = " | ".join(split_names)
    sep = " | ".join(["---:"] * len(split_names))
    lines.append(f"| {header_splits} | total |")
    lines.append(f"| {sep} | ---: |")
    sample_row = " | ".join(str(sample_counts.get(s, 0)) for s in split_names)
    lines.append(f"| {sample_row} | {payload['samples_total']} |")
    lines.append("")
    lines.append("## Per-class annotation counts")
    lines.append("")
    lines.append(
        "| Class | "
        + " | ".join(split_names)
        + " | total |"
    )
    lines.append(
        "|---|"
        + "|".join(["---:"] * len(split_names))
        + "|---:|"
    )
    for cls in CLASSES:
        row = per_class[cls]
        cells = " | ".join(str(row.get(s, 0)) for s in split_names)
        lines.append(f"| {cls} | {cells} | {row.get('total', 0)} |")
    instances_row = " | ".join(
        str(instances_per_split.get(s, 0)) for s in split_names
    )
    lines.append(
        f"| **total instances** | {instances_row} | "
        f"{instances_per_split.get('total', 0)} |"
    )
    lines.append("")

    md_path.write_text("\n".join(lines))
