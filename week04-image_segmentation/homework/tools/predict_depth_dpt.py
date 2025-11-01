from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Set

import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as TF
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "coco"
ANNOTATIONS_ROOT = DATA_ROOT / "annotations"
DEFAULT_IMAGE_DIR = DATA_ROOT / "val2017"
DEFAULT_OUTPUT_DIR = DATA_ROOT / "depth" / "dpt_hybrid"

MODEL_NAME = "Intel/dpt-hybrid-midas"


def iter_images(image_dir: Path, required_files: Iterable[str] | None = None) -> Iterable[Path]:
    if required_files is None:
        files = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for path in files:
            yield path
        return

    required_paths = []
    for file_name in required_files:
        candidate = image_dir / file_name
        if candidate.exists():
            required_paths.append(candidate)
        else:
            # Fallback: some COCO filenames may omit extension in annotations
            # Attempt to append .jpg if missing.
            if not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                alt = image_dir / f"{file_name}.jpg"
                if alt.exists():
                    candidate = alt
                    required_paths.append(candidate)
                    continue
            print(f"[warning] Image not found: {candidate}")

    for path in sorted(required_paths):
        yield path


def load_required_filenames() -> Set[str]:
    mapping = {
        "instances_val2017_train_labeled500.json",
        "instances_val2017_val.json",
        "instances_val2017_test.json",
    }

    required: Set[str] = set()

    for ann_name in mapping:
        ann_path = ANNOTATIONS_ROOT / ann_name
        if not ann_path.exists():
            raise FileNotFoundError(
                f"Required annotation {ann_path} not found. Run prepare_coco.py first."
            )

        with ann_path.open("r") as f:
            data = json.load(f)

        required.update(img["file_name"] for img in data.get("images", []))

    return required


def load_model(device: torch.device):
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    return processor, model


def main() -> None:
    image_dir = DEFAULT_IMAGE_DIR
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "preview"
    visualize = True
    if visualize:
        preview_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, model = load_model(device)

    required_filenames: List[str] | None = None
    required_filenames = sorted(load_required_filenames())

    image_paths = list(iter_images(image_dir, required_filenames))

    if not image_paths:
        print(f"No images found in {image_dir}")
        return

    for idx, path in enumerate(image_paths, 1):
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Resize back to original resolution
        depth = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth = depth.cpu().numpy()
        depth = depth - depth.min()
        depth = depth / (depth.max() + 1e-8)

        np.save(output_dir / f"{path.stem}.npy", depth.astype(np.float32))

        if visualize:
            preview = (depth * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(preview).save(preview_dir / f"{path.stem}.png")

        if idx % 50 == 0 or idx == len(image_paths):
            print(f"Processed {idx}/{len(image_paths)}")


if __name__ == "__main__":
    main()


