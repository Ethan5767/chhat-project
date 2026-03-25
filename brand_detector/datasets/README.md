# Datasets layout

Put training and raw data here so it stays outside `backend/` code and is easy to find.

## `raw_shelf_images/`

**What:** Unlabeled shelf photos (e.g. downloaded from Excel URLs).

**Used by:** `download_shelf_images.py` (default output), `test_yolo.py`, `autolabel_yolo.py` (input).

## `yolo_autolabel_scratch/`

**What:** Default output when you run `autolabel_yolo.py` without `--output-dir`. Pseudo-labeled data from a generic YOLO model before you fix labels in Roboflow/CVAT. You can delete or replace this folder anytime.

## `yolo_cigarette_packs/`

**What:** Active YOLO training set (Ultralytics format): `train/images`, `train/labels`, `valid/`, `test/`, plus `data.yaml`. This folder is the canonical Roboflow export location (rename your `*.yolov11` unzip folder to `yolo_cigarette_packs` and fix `data.yaml` paths to `path: .` + `train: train/images`, etc.).

**Used by:** `run_train.py`, `train_yolo.py`, `yolo detect train ... data=../datasets/yolo_cigarette_packs/data.yaml`.

## Adding a new training dataset

1. Create a folder under `datasets/`, e.g. `datasets/my_yolo_dataset/`.
2. Inside it, use the usual structure:

   ```
   my_yolo_dataset/
     data.yaml
     train/images/  train/labels/
     valid/images/  valid/labels/
     test/images/   test/labels/   (optional)
   ```

3. Point `data.yaml` at this folder (`path: .`) and pass its path to training:

   ```powershell
   cd brand_detector\backend
   yolo detect train data=../datasets/my_yolo_dataset/data.yaml ...
   ```

## Training outputs

Weights and logs from `run_train.py` / `train_yolo.py` are written under `brand_detector/runs/yolo/` by default (see scripts).
