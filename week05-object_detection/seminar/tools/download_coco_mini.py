import os
import urllib.request
import zipfile
from pathlib import Path


def download_file(url, dest_path):
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, dest_path)
    print(f"Downloaded to {dest_path}")


def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")


def main():
    data_dir = Path(__file__).parent.parent / "data" / "coco"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    images_url = "http://images.cocodataset.org/zips/val2017.zip"
    annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    
    images_zip = data_dir / "val2017.zip"
    annotations_zip = data_dir / "annotations_trainval2017.zip"
    
    if not (data_dir / "val2017").exists():
        download_file(images_url, images_zip)
        extract_zip(images_zip, data_dir)
        os.remove(images_zip)
        print("Images downloaded and extracted successfully!")
    else:
        print("Images already exist, skipping download.")
    
    if not (data_dir / "annotations" / "instances_val2017.json").exists():
        download_file(annotations_url, annotations_zip)
        extract_zip(annotations_zip, data_dir)
        os.remove(annotations_zip)
        print("Annotations downloaded and extracted successfully!")
    else:
        print("Annotations already exist, skipping download.")
    
    print("\nCOCO val2017 dataset is ready!")
    print(f"Images: {data_dir / 'val2017'}")
    print(f"Annotations: {data_dir / 'annotations' / 'instances_val2017.json'}")


if __name__ == "__main__":
    main()


