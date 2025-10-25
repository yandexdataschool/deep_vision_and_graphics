import json
import random
from pathlib import Path
from collections import defaultdict


def load_annotations(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)


def save_annotations(data, output_path):
    with open(output_path, 'w') as f:
        json.dump(data, f)
    print(f"Saved to {output_path}")


def filter_person_class(coco_data):
    person_category = None
    for cat in coco_data['categories']:
        if cat['name'] == 'person':
            person_category = cat
            break
    
    if not person_category:
        raise ValueError("Person category not found in COCO dataset")
    
    person_id = person_category['id']
    
    person_annotations = [ann for ann in coco_data['annotations'] if ann['category_id'] == person_id]
    
    image_ids_with_person = set(ann['image_id'] for ann in person_annotations)
    person_images = [img for img in coco_data['images'] if img['id'] in image_ids_with_person]
    
    return person_images, person_annotations, person_category


def create_split(images, annotations, split_name):
    image_ids = set(img['id'] for img in images)
    split_annotations = [ann for ann in annotations if ann['image_id'] in image_ids]
    
    return {
        'info': {'description': f'COCO val2017 person only - {split_name}'},
        'licenses': [],
        'images': images,
        'annotations': split_annotations,
        'categories': [{'id': 1, 'name': 'person', 'supercategory': 'person'}]
    }


def main():
    data_dir = Path(__file__).parent.parent / "data" / "coco"
    annotations_dir = data_dir / "annotations"
    
    input_file = annotations_dir / "instances_val2017.json"
    if not input_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {input_file}")
    
    print(f"Loading {input_file}...")
    coco_data = load_annotations(input_file)
    
    print("Filtering person class...")
    person_images, person_annotations, person_category = filter_person_class(coco_data)
    
    print(f"Found {len(person_images)} images with person annotations")
    print(f"Total person annotations: {len(person_annotations)}")
    
    random.seed(42)
    random.shuffle(person_images)
    
    train_size = 2093
    val_size = 300
    mini_train_size = 1000
    test_size = 300
    
    train_images = person_images[:train_size]
    val_images = person_images[train_size:train_size + val_size]
    mini_train_images = person_images[train_size + val_size:train_size + val_size + mini_train_size]
    test_images = person_images[train_size + val_size + mini_train_size:train_size + val_size + mini_train_size + test_size]
    unlabeled_images = person_images[train_size + val_size + mini_train_size + test_size:]
    
    splits = {
        'train': train_images,
        'val': val_images,
        'mini_train': mini_train_images,
        'test': test_images,
        'unlabeled': unlabeled_images
    }
    
    for split_name, split_images in splits.items():
        if len(split_images) > 0:
            split_data = create_split(split_images, person_annotations, split_name)
            output_file = annotations_dir / f"instances_val2017_{split_name}.json"
            save_annotations(split_data, output_file)
            print(f"  {split_name}: {len(split_images)} images, {len(split_data['annotations'])} annotations")
    
    print("\nDataset splits created successfully!")


if __name__ == "__main__":
    main()


