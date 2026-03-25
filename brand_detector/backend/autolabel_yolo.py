import argparse
import shutil
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def iter_images(folder: Path):
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def yolo_xyxy_to_norm(img_w: int, img_h: int, x1: float, y1: float, x2: float, y2: float):
    x1 = max(0.0, min(float(x1), float(img_w - 1)))
    y1 = max(0.0, min(float(y1), float(img_h - 1)))
    x2 = max(1.0, min(float(x2), float(img_w)))
    y2 = max(1.0, min(float(y2), float(img_h)))
    if x2 <= x1 or y2 <= y1:
        return None

    xc = ((x1 + x2) / 2.0) / img_w
    yc = ((y1 + y2) / 2.0) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return xc, yc, w, h


def run_autolabel(
    image_dir: Path,
    output_dir: Path,
    model_path: str,
    conf: float,
    imgsz: int,
    min_box_wh: int,
    copy_images: bool,
) -> None:
    images_out = output_dir / "images" / "train"
    labels_out = output_dir / "labels" / "train"
    ensure_dir(images_out)
    ensure_dir(labels_out)

    model = YOLO(model_path)
    image_paths = list(iter_images(image_dir))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    total = len(image_paths)
    kept_images = 0
    kept_boxes = 0

    for idx, img_path in enumerate(image_paths, start=1):
        with Image.open(img_path) as img:
            img_w, img_h = img.size

        results = model.predict(str(img_path), conf=conf, imgsz=imgsz, verbose=False)
        label_lines = []

        if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes.xyxy.detach().cpu().numpy():
                norm = yolo_xyxy_to_norm(img_w, img_h, box[0], box[1], box[2], box[3])
                if norm is None:
                    continue
                xc, yc, w, h = norm
                if (w * img_w) < min_box_wh or (h * img_h) < min_box_wh:
                    continue
                label_lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        if label_lines:
            kept_images += 1
            kept_boxes += len(label_lines)
            out_img_path = images_out / img_path.name
            out_lbl_path = labels_out / f"{img_path.stem}.txt"

            if copy_images:
                shutil.copy2(img_path, out_img_path)
            else:
                if out_img_path.exists():
                    out_img_path.unlink()
                out_img_path.symlink_to(img_path.resolve())

            out_lbl_path.write_text("\n".join(label_lines), encoding="utf-8")

        print(f"[{idx}/{total}] {img_path.name} -> {len(label_lines)} boxes")

    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve().as_posix()}",
                "train: images/train",
                "val: images/train",
                "names:",
                "  0: cigarette_pack",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("\nAuto-label complete")
    print(f"Images with labels: {kept_images}/{total}")
    print(f"Total boxes: {kept_boxes}")
    print(f"Dataset folder: {output_dir.resolve()}")
    print(f"data.yaml: {data_yaml.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(description="Auto-label shelf images into YOLO format.")
    parser.add_argument("--image-dir", required=True, help="Folder containing raw shelf photos.")
    parser.add_argument(
        "--output-dir",
        default="../datasets/yolo_autolabel_scratch",
        help="Output folder for pseudo-labeled YOLO data (images/train + labels/train).",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="YOLO model/weights to use for pseudo-label generation.",
    )
    parser.add_argument("--conf", type=float, default=0.20, help="Prediction confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size.")
    parser.add_argument(
        "--min-box-wh",
        type=int,
        default=24,
        help="Minimum width/height in pixels to keep a detection box.",
    )
    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        help="Use symlinks instead of copying images (Windows may require admin/dev mode).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_autolabel(
        image_dir=Path(args.image_dir),
        output_dir=Path(args.output_dir),
        model_path=args.model,
        conf=args.conf,
        imgsz=args.imgsz,
        min_box_wh=args.min_box_wh,
        copy_images=not args.no_copy_images,
    )
