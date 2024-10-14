import argparse
from pathlib import Path
import cv2
import numpy as np
import pydicom
import multiprocessing
import tqdm

from utils import create_directories, divide_into_two_digits, normalize, split_dataset_by_patient, get_mongodb_database


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=str, help="Directory containing DICOM data")
    parser.add_argument("output_dir", type=str, help="Output directory for processed images")
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
    collection = db.get_collection("videos")
    
    query = {"data.category.is_valid": {"$in": [0, 1]}}  # 2는 제외
    
    results = collection.find(query)
    data = []
    for document in results:
        filename = document["filename"]
        patient_id = filename.split("/")[0]
        validity = document["data"]["category"]["is_valid"]
        validity = "valid" if validity == 1 else "invalid"
        data.append({"patient_id": patient_id, "filename": filename, "validity": validity})
    
    print(f"Retrieved {len(data)} documents from MongoDB")
    return data


def process_single_dicom(data, image_size, dataset_dir: Path, output_dir: Path, dataset_type, num_frames_for_train: int, num_interval: int):
    """Process a single DICOM file and save frames in the correct split folder."""
    filename = data["filename"]
    validity = data["validity"]

    filepath = dataset_dir / filename
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

    # Sample frames if necessary
    num_frames = len(pixel_array)
    if num_frames > num_interval * num_frames_for_train:
        pixel_array = pixel_array[::num_interval]
        num_frames = len(pixel_array)

    # Save as .jpeg format for each frame in the appropriate folder (train/val/test)
    num_frames = len(pixel_array)

    if num_frames < num_frames_for_train:
        # Handle case where we need to add frames
        frames_to_add = num_frames_for_train - num_frames
        pre_frames, post_frames = divide_into_two_digits(frames_to_add)

        pre_mask = np.zeros([pre_frames, image_size, image_size], dtype=np.uint8)
        post_mask = np.zeros([post_frames, image_size, image_size], dtype=np.uint8)

        image = np.concatenate([pre_mask, pixel_array, post_mask], axis=0)
    else:
        # Handle case where we need to select a subset of frames
        frames_to_add = num_frames - num_frames_for_train
        pre_frames, post_frames = divide_into_two_digits(frames_to_add)

        image = pixel_array[pre_frames : num_frames - post_frames]

    # Validate final image shape
    assert (
        len(image) == num_frames_for_train
    ), f"Expected {num_frames_for_train} frames, got {len(image)}"

    np.save(image_save_path, image[None])  # image.shape: (C, T, H, W)


def main():
    args = parse_args()

    # Output directory setup
    dataset_dir = Path(args.dataset_dir)
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
    all_data = [(d, args.image_size, dataset_dir, output_dir, 'train', args.num_frames_for_train, args.num_interval) for d in train_data] + \
               [(d, args.image_size, dataset_dir, output_dir, 'val', args.num_frames_for_train, args.num_interval) for d in val_data] + \
               [(d, args.image_size, dataset_dir, output_dir, 'test', args.num_frames_for_train, args.num_interval) for d in test_data]

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

