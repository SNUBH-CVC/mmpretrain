"""
python tools/dataset_converters/cag_frame_selection.py \
    /mnt/nas/snubhcvc/raw/cag_ccta_1yr_all/data \
    /mnt/nas/snubhcvc/raw/cpacs \
    --output_dir ./data/cag_frame_selection
    --num_processes 4
"""

import argparse
import multiprocessing
from pathlib import Path

import cv2
import numpy as np
import pydicom
import tqdm

from mmpretrain_utils.preprocess import split_dataset_by_patient, normalize
from mmpretrain_utils.utils import create_directories, get_mongodb_database


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dirs", nargs="+", type=str, help="Directory containing DICOM data")
    parser.add_argument("--output_dir", type=str, help="Output directory for processed images")
    parser.add_argument("--image_size", type=int, default=512, help="Size to which images will be resized")
    parser.add_argument("--num_processes", type=int, default=None, help="Number of processes to use for multiprocessing")
    parser.add_argument("--test_size", type=float, default=0.1, help="Fraction of the data to use for testing")
    parser.add_argument("--val_size", type=float, default=0.1, help="Fraction of the data to use for validation")
    return parser.parse_args()


def get_data_from_mongodb():
    """Retrieve relevant DICOM data from MongoDB"""
    db = get_mongodb_database()
    collection = db.get_collection("frames")
    
    query = {
        "$and": [
            {"data": {"$elemMatch": {"category.qca_valid": 0}}},
            {"$expr": {"$eq": [{"$size": "$data"}, 1]}},
        ]
    }
    
    results = collection.find(query)
    data = []
    for document in results:
        filename = document["filename"]
        patient_id = filename.split("/")[0]
        frame_idx = document["data"][0]["index"]
        data.append({"patient_id": patient_id, "filename": filename, "frame_idx": frame_idx})
    
    print(f"Retrieved {len(data)} documents from MongoDB")
    return data


def process_single_dicom(data, image_size, dataset_dirs: list[Path], output_dir: Path, dataset_type: str):
    """Process a single DICOM file and save frames in the correct split folder."""
    filename = data["filename"]
    frame_idx = data["frame_idx"]

    filepath = dataset_dirs[0] / filename
    if not filepath.exists():
        filepath = dataset_dirs[1] / filename
        if not filepath.exists():
            print(f"No such file: {filepath}")
            return
    patient_id = filepath.parents[2].name
    study_date = filepath.parents[1].name

    basename_wo_ext = filepath.stem
    uid = f"{patient_id}_{study_date}_{basename_wo_ext}"
    save_dir = output_dir / dataset_type 
    image_save_path = save_dir / f"{uid}_{frame_idx}.npy"
    if image_save_path.exists():
        print(f"File already exists: {image_save_path}")
        return

    dcm = pydicom.dcmread(filepath)
    pixel_array = dcm.pixel_array
    if pixel_array.dtype != np.uint8:
        pixel_array = (normalize(pixel_array) * 255).astype(np.uint8)

    # Ensure 3D pixel array (frames, height, width)
    assert len(pixel_array.shape) == 3, "Expected 3D pixel array"

    # Resize
    if pixel_array.shape[1] != image_size or pixel_array.shape[2] != image_size:
        pixel_array = np.stack(
            [cv2.resize(frame, (image_size, image_size)) for frame in pixel_array]
        )

    np.save(image_save_path, pixel_array[None])  # pixel_array.shape: (C, T, H, W)


def main():
    args = parse_args()

    # Output directory setup
    dataset_dirs = [Path(dir) for dir in args.dataset_dirs]
    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Get data from MongoDB
    data = get_data_from_mongodb()

    # Split dataset into train, val, and test sets
    train_data, val_data, test_data = split_dataset_by_patient(data, args.test_size, args.val_size)

    # Create directories for train, val, and test splits with class folders
    create_directories(Path(args.output_dir), ['train', 'val', 'test'])

    # Combine all data into a single list with dataset type
    all_data = [(d, args.image_size, dataset_dirs, output_dir, 'train') for d in train_data] + \
               [(d, args.image_size, dataset_dirs, output_dir, 'val') for d in val_data] + \
               [(d, args.image_size, dataset_dirs, output_dir, 'test') for d in test_data]

    # Multiprocessing for parallel processing of all DICOM files
    with multiprocessing.Pool(args.num_processes) as pool:
        pool.starmap(
            process_single_dicom, 
            tqdm.tqdm(
                all_data,
                total=len(all_data),
                desc="Processing all data"
            )
        )


if __name__ == "__main__":
    main()
