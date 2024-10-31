import argparse
from pathlib import Path

import tqdm
import pandas as pd
import cv2
import numpy as np
import pydicom
import multiprocessing

from utils import split_dataset_by_patient, normalize, create_directories, divide_into_two_digits, get_mongodb_database


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
    parser.add_argument("--K", type=int, default=500, help="Number of samples to take from each class(angle, coronary)")
    return parser.parse_args()


def angle_to_class(alpha, beta):
    if alpha <= -10:
        prefix = "RAO"
    elif -10 < alpha < 10:
        prefix = "AP"
    else:
        prefix = "LAO"

    if beta <= -10:
        postfix = "CAUD"
    elif -10 < beta < 10:
        postfix = "AP"
    else:
        postfix = "CRAN"

    if postfix == "AP":
        cls = prefix
    else:
        cls = f"{prefix}_{postfix}"
    return cls


def get_meta_data_from_mongodb():
    db = get_mongodb_database()
    collection = db.get_collection("meta_xa")
    meta_xa_documents = collection.find()
    meta_xa_dict = {}
    for doc in meta_xa_documents:
        filename = doc.pop("filename")
        meta_xa_dict.update({filename: doc})
    return meta_xa_dict


def sample_data(data, K=500):
    # Group by 'coronary_cls' and 'angle_cls'
    grouped = pd.DataFrame(data).groupby(['angle_cls', 'coronary_cls'])
    
    # Sample up to K data points per group
    sampled_data = grouped.apply(lambda x: x.sample(min(len(x), K), random_state=42)).reset_index(drop=True)
    
    return sampled_data.to_dict(orient='records')


def get_data_from_mongodb():
    """Retrieve relevant DICOM data from MongoDB"""
    db = get_mongodb_database()
    videos_collection = db.get_collection("videos")
    meta_xa_dict = get_meta_data_from_mongodb()
    
    query = {
        "$and": [
            {"data.category.left_right": {"$in": [0, 1]}},
        ]
    }
    
    results = videos_collection.find(query)
    data = []
    for document in results:
        filename = document["filename"]
        meta = meta_xa_dict.get(filename)
        alpha = meta["positioner_primary_angle"]
        beta = meta["positioner_secondary_angle"]
        number_of_frames = meta["number_of_frames"]
        if alpha is None or beta is None or number_of_frames is None or number_of_frames == 1:
            continue
        angle_cls = angle_to_class(alpha, beta)
        patient_id = filename.split("/")[0]
        coronary_cls = document["data"]["category"]["left_right"]
        coronary_cls = "left" if coronary_cls == 0 else "right"
        data.append({"patient_id": patient_id, "filename": filename, "coronary_cls": coronary_cls, "angle_cls": angle_cls})

    print("Before sampling", pd.DataFrame(data).groupby(["angle_cls", "coronary_cls"])[["filename"]].count().to_markdown())
    sampled_data = sample_data(data)
    print("After sampling", pd.DataFrame(sampled_data).groupby(["angle_cls", "coronary_cls"])[["filename"]].count().to_markdown())
    print(f"Retrieved {len(sampled_data)} documents from MongoDB")
    return sampled_data


def process_single_dicom(data, image_size, dataset_dir: Path, output_dir: Path, dataset_type, num_frames_for_train: int, num_interval: int):
    """Process a single DICOM file and save frames in the correct split folder."""
    filename = data["filename"]
    coronary_cls = data["coronary_cls"]

    filepath = dataset_dir / filename
    if not filepath.exists():
        print(f"No such file: {filepath}")
        return
    patient_id = filepath.parents[2].name
    study_date = filepath.parents[1].name

    basename_wo_ext = filepath.stem
    uid = f"{patient_id}_{study_date}_{basename_wo_ext}"
    class_folder = f"class_{coronary_cls}"  # Example class name based on coronary_cls
    save_dir = output_dir / dataset_type / class_folder
    image_save_path = save_dir / f"{uid}.npy"
    if image_save_path.exists():
        print(f"File already exists: {image_save_path}")
        return

    try:
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
    except Exception as e:
        print(f"Error processing {filepath}: {e}")


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
    classes = ['left', 'right']

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
