"""
python tools/dataset_converters/cag_video_classification.py \
    /mnt/nas/snubhcvc/raw/cag_ccta_1yr_all/data \
    /mnt/nas/snubhcvc/raw/cpacs \
    --output_dir ./data/cag_video_classification
"""

import argparse
from pathlib import Path
import numpy as np
import multiprocessing
import tqdm

from mmpretrain_utils.utils import create_directories, get_mongodb_database
from mmpretrain_utils.preprocess import load_and_process_dicom, resample_frames_evenly, split_dataset_by_patient


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dirs", nargs="+", type=str, help="List of directories containing DICOM data")
    parser.add_argument("--output_dir", type=str, help="Output directory for processed images")
    parser.add_argument("--image_size", type=int, default=512, help="Size to which images will be resized")
    parser.add_argument("--num_frames_for_train", type=int, default=60, help="Number of frames to use for training")
    parser.add_argument("--num_interval", type=int, default=1, help="Interval for frame sampling if pixel array length is too large")
    parser.add_argument("--num_processes", type=int, default=None, help="Number of processes to use for multiprocessing")
    parser.add_argument("--test_size", type=float, default=0.1, help="Fraction of the data to use for testing")
    parser.add_argument("--val_size", type=float, default=0.1, help="Fraction of the data to use for validation")
    return parser.parse_args()


def get_data_from_mongodb():
    """Retrieve relevant DICOM data from MongoDB"""
    db = get_mongodb_database()
    videos_collection = db.get_collection("videos")

    query = {
        "$or": [
            {
                "data.category.is_valid": {"$in": [1]},
                "data.category.left_right": {"$exists": True}
            },
            {
                "data.category.is_valid": {"$in": [0]},
            }
        ]
    }
    results = videos_collection.find(query)
    data = []
    for document in results:
        filename = document["filename"]
        patient_id = filename.split("/")[0]
        validity = document["data"]["category"]["is_valid"]
        validity = "valid" if validity == 1 else "invalid"
        data.append({"patient_id": patient_id, "filename": filename, "validity": validity})
    
    print(f"Retrieved {len(data)} documents from MongoDB")
    return data


def process_single_dicom(data, image_size, dataset_dirs: list[Path], output_dir: Path, dataset_type, num_frames_for_train: int, num_interval: int):
    """Process a single DICOM file and save frames in the correct split folder."""
    filename = data["filename"]
    validity = data["validity"]

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
    class_folder = f"class_{validity}"  # Example class name based on coronary_cls
    save_dir = output_dir / dataset_type / class_folder
    image_save_path = save_dir / f"{uid}.npy"
    if image_save_path.exists():
        print(f"File already exists: {image_save_path}")
        return

    try:
        image = load_and_process_dicom(filepath, image_size)
        if image is None:
            print(f"Error loading and processing DICOM file {filepath}")
            return
        image = resample_frames_evenly(image, num_frames_for_train, image_size)

        # Validate final image shape
        assert (
            len(image) == num_frames_for_train
        ), f"Expected {num_frames_for_train} frames, got {len(image)}"

        np.save(image_save_path, image[None])  # image.shape: (C, T, H, W)
    except Exception as e:
        print(f"Error processing {filepath}: {e}")


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


    # Define classes based on left/right coronary classifications
    classes = ['valid', 'invalid']

    # Create directories for train, val, and test splits with class folders
    create_directories(Path(args.output_dir), ['train', 'val', 'test'], classes)

    # Combine all data into a single list with dataset type
    all_data = [(d, args.image_size, dataset_dirs, output_dir, 'train', args.num_frames_for_train, args.num_interval) for d in train_data] + \
               [(d, args.image_size, dataset_dirs, output_dir, 'val', args.num_frames_for_train, args.num_interval) for d in val_data] + \
               [(d, args.image_size, dataset_dirs, output_dir, 'test', args.num_frames_for_train, args.num_interval) for d in test_data]

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
