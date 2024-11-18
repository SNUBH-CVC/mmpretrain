from pathlib import Path

import pymongo


def get_mongodb_database():
    client = pymongo.MongoClient("mongodb://172.30.1.46:27017")
    return client.get_database("snubhcvc")


def create_directories(output_dir: Path, dataset_types, classes):
    """Create directories for train/val/test and class_left/class_right."""
    for dataset_type in dataset_types:
        for cls in classes:
            save_dir = output_dir / dataset_type / f"class_{cls}"
            save_dir.mkdir(parents=True, exist_ok=True)

