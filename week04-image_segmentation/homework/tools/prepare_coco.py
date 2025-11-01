import json
import random
from pathlib import Path


def load_annotations(json_path: Path) -> dict:
    with json_path.open("r") as f:
        return json.load(f)


def save_annotations(data: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(data, f)
    print(f"Saved {output_path}")


def create_split(images, annotations, categories, split_name: str) -> dict:
    image_ids = {img["id"] for img in images}
    split_annotations = [ann for ann in annotations if ann["image_id"] in image_ids]

    return {
        "info": {"description": f"COCO val2017 person subset - {split_name}"},
        "licenses": [],
        "images": images,
        "annotations": split_annotations,
        "categories": categories,
    }


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data" / "coco"
    annotations_dir = data_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    base_ann = annotations_dir / "instances_val2017.json"
    if not base_ann.exists():
        raise FileNotFoundError(
            f"Base COCO annotations not found at {base_ann}. Run download_coco_mini.py first."
        )

    print(f"Loading {base_ann}...")
    coco_data = load_annotations(base_ann)

    categories = [cat for cat in coco_data["categories"] if cat["name"] == "person"]
    if not categories:
        raise ValueError("Expected 'person' category in COCO annotations.")

    person_id = categories[0]["id"]
    person_annotations = [ann for ann in coco_data["annotations"] if ann["category_id"] == person_id]
    image_ids_with_person = {ann["image_id"] for ann in person_annotations}
    person_images = [img for img in coco_data["images"] if img["id"] in image_ids_with_person]

    print(f"Total person images: {len(person_images)}")
    print(f"Total person annotations: {len(person_annotations)}")

    random.seed(42)
    random.shuffle(person_images)

    train_count = 500
    val_count = 300
    test_count = 300

    train_images = person_images[:train_count]
    val_images = person_images[train_count : train_count + val_count]
    test_images = person_images[train_count + val_count : train_count + val_count + test_count]

    splits = {
        "instances_val2017_train_labeled500.json": create_split(
            train_images, person_annotations, categories, "train_labeled500"
        ),
        "instances_val2017_val.json": create_split(val_images, person_annotations, categories, "val"),
        "instances_val2017_test.json": create_split(test_images, person_annotations, categories, "test"),
    }

    for filename, content in splits.items():
        target = annotations_dir / filename
        save_annotations(content, target)
        print(
            f"  {filename}: {len(content['images'])} images, {len(content['annotations'])} annotations"
        )

    print("\nCOCO splits ready!")


if __name__ == "__main__":
    main()

