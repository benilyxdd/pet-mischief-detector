"""Turn ``outputs/eval/<split>/`` artifacts into a report-ready summary.

Reads the JSON metrics + confusion matrix CSV that
:func:`src.evaluation.detector_eval.evaluate_detector` persists, produces:

* ``outputs/figures/per_class_ap.png`` -- sorted per-class AP bar chart.
* ``outputs/eval/<split>/analysis.md`` -- markdown summary with the
  headline metrics, per-class table, and top confusion pairs. This is
  meant to be quoted wholesale in the individual report.

The script is deliberately self-contained so it can be re-run without
retraining or re-evaluating.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # no-display backend; safe for headless boxes
import matplotlib.pyplot as plt

from src.constants.classes import CLASSES
from src.constants.paths import EVAL_DIR, FIGURES_DIR
from src.utils.io import load_json
from src.utils.logging import get_logger
from src.utils.paths import ensure_dir

logger = get_logger(__name__)

_TOP_CONFUSIONS = 5


def analyze_evaluation(
    split: str = "test",
    eval_dir: Path | None = None,
    figures_dir: Path | None = None,
) -> Path:
    """Build the analysis artifacts for the given eval split.

    Returns the path to the written markdown file.
    """
    split_dir = (eval_dir or EVAL_DIR) / split
    metrics_path = split_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} not found. Run option 3 'Evaluate model' first."
        )

    report = load_json(metrics_path)

    fig_dir = ensure_dir(figures_dir or FIGURES_DIR)
    chart_path = _plot_per_class_ap(report, fig_dir)

    cm_csv = report.get("confusion_matrix_csv")
    confusions = _top_confusions(Path(cm_csv)) if cm_csv and Path(cm_csv).exists() else []

    md_path = split_dir / "analysis.md"
    _write_markdown(report, chart_path, confusions, md_path)
    logger.info("Wrote analysis markdown -> %s", md_path)
    logger.info("Wrote per-class AP chart -> %s", chart_path)
    return md_path


# ---------------------------------------------------------------------------
# Per-class AP chart
# ---------------------------------------------------------------------------


def _plot_per_class_ap(report: dict[str, Any], fig_dir: Path) -> Path:
    per_class: dict[str, dict[str, Any]] = report.get("per_class", {})
    items = [
        (cls, float(vals.get("AP50-95") or 0.0))
        for cls, vals in per_class.items()
    ]
    items.sort(key=lambda x: x[1])

    names = [n for n, _ in items]
    values = [v for _, v in items]
    mean_ap = float(report.get("mAP50-95") or 0.0)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(names, values)
    ax.axvline(mean_ap, linestyle="--", linewidth=1, color="tab:red",
               label=f"mean mAP50-95 = {mean_ap:.3f}")
    ax.set_xlabel("AP@[0.5:0.95]")
    ax.set_xlim(0.0, max(1.0, max(values) + 0.05) if values else 1.0)
    ax.set_title("Per-class AP@[0.5:0.95] -- test split")
    ax.legend(loc="lower right")
    for bar, v in zip(bars, values):
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}", va="center", fontsize=9)
    fig.tight_layout()

    out_path = fig_dir / "per_class_ap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Confusion matrix -> top confused pairs
# ---------------------------------------------------------------------------


def _top_confusions(csv_path: Path) -> list[dict[str, Any]]:
    """Return the top off-diagonal confusion pairs from the CSV.

    The CSV layout matches Ultralytics' internal confusion matrix:
    rows = predicted class (+ background), cols = true class (+ background).
    We report (true, pred, count) tuples for the largest off-diagonal cells,
    skipping the background column / row (those are just FP / FN buckets
    that aren't class-vs-class confusion).
    """
    try:
        rows = _read_csv(csv_path)
    except Exception as e:
        logger.warning("Could not parse confusion matrix CSV %s: %s", csv_path, e)
        return []
    if not rows:
        return []

    header = rows[0][1:]
    data = rows[1:]

    name_to_idx = {n: i for i, n in enumerate(header)}

    confusions: list[dict[str, Any]] = []
    for row in data:
        if not row:
            continue
        pred_name = row[0]
        if pred_name == "background":
            continue
        for col_name, cell in zip(header, row[1:]):
            if col_name == "background" or col_name == pred_name:
                continue
            try:
                count = float(cell)
            except ValueError:
                continue
            if count <= 0:
                continue
            confusions.append({
                "true": col_name,
                "predicted": pred_name,
                "count": count,
            })

    confusions.sort(key=lambda d: d["count"], reverse=True)
    return confusions[:_TOP_CONFUSIONS]


def _read_csv(path: Path) -> list[list[str]]:
    import csv

    with open(path, "r") as f:
        return [row for row in csv.reader(f)]


# ---------------------------------------------------------------------------
# Markdown writer
# ---------------------------------------------------------------------------


def _write_markdown(
    report: dict[str, Any],
    chart_path: Path,
    confusions: list[dict[str, Any]],
    md_path: Path,
) -> None:
    ensure_dir(md_path.parent)

    lines: list[str] = []
    lines.append("# Detector evaluation summary")
    lines.append("")
    lines.append(f"- **Split**: `{report.get('split', 'test')}`")
    lines.append(f"- **Weights**: `{report.get('weights', '')}`")
    lines.append(f"- **mAP@0.5**: {_fmt(report.get('mAP50'))}")
    lines.append(f"- **mAP@[0.5:0.95]**: {_fmt(report.get('mAP50-95'))}")
    lines.append(f"- **mAP@0.75**: {_fmt(report.get('mAP75'))}")
    lines.append(f"- **Mean precision**: {_fmt(report.get('precision_mean'))}")
    lines.append(f"- **Mean recall**: {_fmt(report.get('recall_mean'))}")
    if report.get("device"):
        lines.append(f"- **Device**: `{report.get('device')}`")
    inference_ms = report.get("inference_ms_per_image")
    total_ms = report.get("total_ms_per_image")
    fps = report.get("fps")
    if inference_ms or total_ms:
        lines.append(
            "- **Speed**: "
            f"inference {_fmt(inference_ms)} ms/image, "
            f"total {_fmt(total_ms)} ms/image"
            + (f", ~{_fmt(fps)} FPS" if fps else "")
        )
    eval_cfg = report.get("eval_config") or {}
    if eval_cfg:
        lines.append(f"- **Eval config**: `{eval_cfg}`")
    lines.append("")

    import os

    rel_chart = os.path.relpath(chart_path.resolve(), md_path.resolve().parent)
    lines.append("## Per-class AP")
    lines.append("")
    lines.append(f"![Per-class AP]({rel_chart})")
    lines.append("")
    lines.append("| Class | AP@[0.5:0.95] | Precision | Recall | F1 |")
    lines.append("|---|---:|---:|---:|---:|")
    for cls in CLASSES:
        vals = report.get("per_class", {}).get(cls, {})
        lines.append(
            f"| {cls} "
            f"| {_fmt(vals.get('AP50-95'))} "
            f"| {_fmt(vals.get('precision'))} "
            f"| {_fmt(vals.get('recall'))} "
            f"| {_fmt(vals.get('f1'))} |"
        )
    lines.append("")

    if confusions:
        lines.append("## Top confusion pairs")
        lines.append("")
        lines.append("Off-diagonal cells in the normalised confusion matrix ")
        lines.append("(Ultralytics orientation: row = predicted, column = true).")
        lines.append("")
        lines.append("| True class | Predicted as | Count (normalised) |")
        lines.append("|---|---|---:|")
        for row in confusions:
            lines.append(
                f"| {row['true']} | {row['predicted']} | {row['count']:.3f} |"
            )
        lines.append("")
        lines.append(
            "_Interpretation hint_: if a pet class is frequently confused with "
            "a furniture class (e.g. `dog` <-> `couch`), raising `cls` in "
            "`configs/train.yaml` from 0.5 to ~0.7 typically helps."
        )
        lines.append("")
    else:
        lines.append("## Top confusion pairs")
        lines.append("")
        lines.append(
            "_Confusion matrix CSV not found; skip this section or re-run "
            "evaluation to generate `confusion_matrix.csv`._"
        )
        lines.append("")

    md_path.write_text("\n".join(lines))


def _fmt(x: Any) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return str(x)
