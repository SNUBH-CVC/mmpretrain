import csv
from pathlib import Path

import numpy as np
import pymongo


def get_mongodb_database():
    client = pymongo.MongoClient("mongodb://172.30.1.46:27017")
    return client.get_database("snubhcvc")


def create_directories(output_dir: Path, dataset_types, classes=None):
    """Create directories for train/val/test and class_left/class_right."""
    for dataset_type in dataset_types:
        if classes is None:
            save_dir = output_dir / dataset_type
        else:
            for cls in classes:
                save_dir = output_dir / dataset_type / f"class_{cls}"
        save_dir.mkdir(parents=True, exist_ok=True)


def get_kth_largest_rank(arr, index):
    """
    Returns the rank (1-based) of the element at the specified index 
    in terms of largest values in the array.
    
    Parameters:
    arr (np.ndarray): The input array.
    index (int): The index of the element whose rank is to be found.
    
    Returns:
    int: The rank of the element at the given index.
    """
    # Get the indices that would sort the array in descending order
    sorted_indices = np.argsort(arr)[::-1]
    
    # Find the 1-based rank of the specified index
    k = np.where(sorted_indices == index)[0][0] + 1
    
    return k


class CSVManager:
    def __init__(self, output_path, headers):
        self.output_path = Path(output_path)
        self.headers = headers
        if self.output_path.exists():
            self.processed_files = self._load_processed_files()
        else:
            # Create directory if it doesn't exist
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.processed_files = {}
            # Write headers to the new file
            self._write_headers()
        self.csv_file = None  # File object
        self.csv_writer = None

    def _write_headers(self):
        """Write headers to the CSV file."""
        with open(self.output_path, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(self.headers)

    def _load_processed_files(self):
        """Load already processed entries from the CSV file."""
        processed_files = {}
        with open(self.output_path, mode='r', newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader, None)  # Skip header
            for row in csv_reader:
                key = f"{row[0]}_{row[1]}_{row[2]}"
                processed_files[key] = True
        return processed_files

    def __enter__(self):
        """Enter the runtime context and open the file for appending."""
        self.csv_file = open(self.output_path, mode='a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the runtime context and close the file."""
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

    def is_processed(self, key):
        """Check if a file is already processed."""
        return self.processed_files.get(key, False)

    def write_row(self, row):
        """Write a row to the CSV file."""
        if self.csv_writer:
            self.csv_writer.writerow(row)
            self.csv_file.flush()
        else:
            raise RuntimeError("CSV file is not open. Use `with` statement to manage the file context.")
