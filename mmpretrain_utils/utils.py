from pathlib import Path

import pymongo
from sklearn.model_selection import train_test_split


def get_mongodb_database():
    client = pymongo.MongoClient("mongodb://localhost:27017")
    return client.get_database("snubhcvc")


def create_directories(output_dir: Path, dataset_types, classes):
    """Create directories for train/val/test and class_left/class_right."""
    for dataset_type in dataset_types:
        for cls in classes:
            save_dir = output_dir / dataset_type / f"class_{cls}"
            save_dir.mkdir(parents=True, exist_ok=True)

