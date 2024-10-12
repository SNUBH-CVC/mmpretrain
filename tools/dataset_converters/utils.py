import pymongo
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split


def get_mongodb_database():
    client = pymongo.MongoClient("mongodb://localhost:27017")
    return client.get_database("snubhcvc")


def normalize(pixel_array):
    """Normalize pixel values to 0-1 range."""
    min_val = np.min(pixel_array)
    max_val = np.max(pixel_array)
    return (pixel_array - min_val) / (max_val - min_val)


def divide_into_two_digits(number):
    first_digit = (number + 1) // 2
    second_digit = number - first_digit
    return first_digit, second_digit


def create_directories(output_dir: Path, dataset_types, classes):
    """Create directories for train/val/test and class_left/class_right."""
    for dataset_type in dataset_types:
        for cls in classes:
            save_dir = output_dir / dataset_type / f"class_{cls}"
            save_dir.mkdir(parents=True, exist_ok=True)


def split_dataset_by_patient(data, test_size, val_size):
    """Split the dataset into train, val, and test sets based on patient IDs."""
    # Collect unique patient IDs
    patient_ids = list(set([d['patient_id'] for d in data]))
    
    # Split patient IDs into train+val and test
    train_val_ids, test_ids = train_test_split(patient_ids, test_size=test_size, random_state=42)
    
    # Adjust val_size to account for the reduced dataset after the test split
    val_size_adjusted = val_size / (1 - test_size)
    
    # Split train_val_ids into train and val
    train_ids, val_ids = train_test_split(train_val_ids, test_size=val_size_adjusted, random_state=42)
    
    # Assign data entries to splits based on patient IDs
    train_data = [d for d in data if d['patient_id'] in train_ids]
    val_data = [d for d in data if d['patient_id'] in val_ids]
    test_data = [d for d in data if d['patient_id'] in test_ids]
    
    print(f"Total patients: {len(patient_ids)}")
    print(f"Train patients: {len(train_ids)}, Data entries: {len(train_data)}")
    print(f"Validation patients: {len(val_ids)}, Data entries: {len(val_data)}")
    print(f"Test patients: {len(test_ids)}, Data entries: {len(test_data)}")
    
    return train_data, val_data, test_data
