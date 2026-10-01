# Pet Mischief Detector

Warns when a pet is near a household object it might disturb.

## What it does

The pipeline filters a COCO subset into YOLO labels. The CLI is the numbered menu in the last section. The spatial scorer ranks pet–object pairs by overlap and distance into tiered warnings. Risk tiers and warning text live in `configs/mischief.yaml`.

## Run

From the repository root, with Python 3.12 or newer and uv already installed:

```bash
uv sync
uv run python main.py
```

Those commands open the CLI. That is the only finish this README promises.

## Menu

The CLI is a numbered menu. It opens from the commands above and returns after each action until you exit. None of these actions is promised to finish.

- **Prepare** downloads the uncapped COCO subset, then filters, splits, and exports YOLO labels.
- **Train** needs that dataset. It fine-tunes YOLOv8s for 150 epochs at image size 768 and batch 12, sized for one CUDA GPU. That schedule lives in `configs/train.yaml`. That run is how a checkpoint appears.
- **Resume** continues an interrupted training run from the last checkpoint.
- **Evaluate** checks a trained checkpoint against the prepared dataset.
- **Analyze** writes a chart and a short written summary from an evaluation that already exists.
- **Infer** runs the detector on an image path you supply (the menu labels this "Run mischief detector on an image"). It needs a trained checkpoint. The clone contains no sample image.
