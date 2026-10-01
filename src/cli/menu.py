"""Numbered menu.

Opens from ``uv sync`` and ``uv run python main.py`` and returns to the menu
until Exit. Exceptions are caught and printed so one broken option cannot
crash the whole session.
"""
from __future__ import annotations

import traceback
from pathlib import Path


MENU_TITLE = "===== Pet Mischief Detector ====="
MENU_ITEMS: tuple[tuple[str, str, str | None], ...] = (
    ("1", "Prepare dataset    (download COCO -> filter -> split -> YOLO export)", "_prepare"),
    ("2", "Train model        (YOLOv8s fine-tune on curated set)", "_train"),
    ("R", "Resume last run    (pick up from runs/.../last.pt after OOM/Ctrl+C)", "_resume"),
    ("3", "Evaluate model     (mAP@0.5, mAP@[0.5:0.95], per-class AP, TTA)", "_evaluate"),
    ("4", "Analyze evaluation (per-class chart + confusion matrix summary)", "_analyze"),
    ("5", "Run mischief detector on an image", "_infer"),
    ("0", "Exit", None),
)


def _prepare() -> None:
    from src.data.prepare import prepare_dataset

    prepare_dataset()


def _train() -> None:
    from src.training.train import run_training

    run_training()


def _resume() -> None:
    from src.training.train import resume_training

    resume_training()


def _evaluate() -> None:
    from src.evaluation.detector_eval import evaluate_detector

    evaluate_detector()


def _analyze() -> None:
    from src.evaluation.analyze import analyze_evaluation

    analyze_evaluation()


def _infer() -> None:
    from src.inference.infer import run_inference

    image = input("Image path: ").strip()
    if not image:
        print("No image path provided.")
        return
    out = input(
        "Output path [blank = outputs/predictions/<image>_annotated.jpg]: "
    ).strip()
    out_path = Path(out) if out else None
    run_inference(image, out_path)


def _print_menu() -> None:
    print()
    print(MENU_TITLE)
    for key, label, _ in MENU_ITEMS:
        print(f"  {key}) {label}")


def run() -> None:
    handlers = {key.upper(): fn_name for key, _, fn_name in MENU_ITEMS}
    while True:
        _print_menu()
        try:
            choice = input("Select an option: ").strip().upper()
        except EOFError:
            print()
            return

        if choice not in handlers:
            print(f"Invalid choice: {choice!r}")
            continue

        fn_name = handlers[choice]
        if fn_name is None:
            print("Bye.")
            return

        fn = globals()[fn_name]
        try:
            fn()
        except KeyboardInterrupt:
            print("\nInterrupted. Returning to main menu.")
        except Exception as e:  # noqa: BLE001 - safety net so one option cannot crash the session
            print(f"\n[ERROR] {e}")
            traceback.print_exc()
            print("Returning to main menu.")
