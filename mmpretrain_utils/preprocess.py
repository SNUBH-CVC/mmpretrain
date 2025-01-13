from collections import defaultdict

import numpy as np
import pydicom
import cv2
from sklearn.model_selection import train_test_split


def normalize(pixel_array):
    """Normalize pixel values to 0-1 range."""
    min_val = np.min(pixel_array)
    max_val = np.max(pixel_array)
    return (pixel_array - min_val) / (max_val - min_val)


def divide_into_two_digits(number):
    first_digit = (number + 1) // 2
    second_digit = number - first_digit
    return first_digit, second_digit


def split_dataset_by_patient(data, test_size, val_size, random_state=42):
    """Split the dataset into train, val, and test sets based on patient IDs."""
    # Collect unique patient IDs
    patient_ids_dict = defaultdict(list)
    for d in data:
        patient_ids_dict[d['patient_id']].append(d)
    
    # Split patient IDs into train+val and test
    train_val_ids, test_ids = train_test_split(list(patient_ids_dict.keys()), test_size=test_size, random_state=random_state)
    
    # Adjust val_size to account for the reduced dataset after the test split
    val_size_adjusted = val_size / (1 - test_size)
    
    # Split train_val_ids into train and val
    train_ids, val_ids = train_test_split(train_val_ids, test_size=val_size_adjusted, random_state=random_state)
    
    # Assign data entries to splits based on patient IDs
    train_data = []
    val_data = []
    test_data = []
    for patient_id in train_ids:
        train_data.extend(patient_ids_dict[patient_id])
    for patient_id in val_ids:
        val_data.extend(patient_ids_dict[patient_id])
    for patient_id in test_ids:
        test_data.extend(patient_ids_dict[patient_id])
    
    print(f"Total patients: {len(patient_ids_dict)}")
    print(f"Train patients: {len(train_ids)}, Data entries: {len(train_data)}")
    print(f"Validation patients: {len(val_ids)}, Data entries: {len(val_data)}")
    print(f"Test patients: {len(test_ids)}, Data entries: {len(test_data)}")
    
    return train_data, val_data, test_data


def load_and_process_dicom(filepath, image_size):
    """
    Load a DICOM file and process it into a numpy array.
    
    Parameters:
    - filepath: Path to the DICOM file.
    - image_size: Desired size to which the image will be resized.
    
    Returns:
    - A numpy array of the processed image.
    """
    try:
        # Read the DICOM file
        dcm = pydicom.dcmread(filepath)
        pixel_array = dcm.pixel_array
        
        # Normalize the pixel array if necessary
        if pixel_array.dtype != np.uint8:
            pixel_array = (normalize(pixel_array) * 255).astype(np.uint8)
        
        # Ensure the pixel array is 3D (frames, height, width)
        assert len(pixel_array.shape) == 3, "Expected 3D pixel array"
        
        # Resize the image if necessary
        if pixel_array.shape[1] != image_size or pixel_array.shape[2] != image_size:
            pixel_array = np.stack(
                [cv2.resize(frame, (image_size, image_size)) for frame in pixel_array]
            )
        
        return pixel_array
    
    except Exception as e:
        print(f"Error loading and processing DICOM file {filepath}: {e}")
        return None


def resample_mid_frames(pixel_array, num_frames_for_train, image_size):
    """
    Adjust the number of frames in the pixel array to match num_frames_for_train.
    
    Parameters:
    - pixel_array: The original pixel array from the DICOM file.
    - num_frames_for_train: The desired number of frames.
    - image_size: The size of the image (height and width).
    
    Returns:
    - A numpy array with the adjusted number of frames.
    """
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
    
    return image


def resample_frames_evenly(pixel_array, num_frames_for_train, image_size):
    """
    Adjust the number of frames in the pixel array to match num_frames_for_train.
    
    If num_frames < num_frames_for_train:
        - Pad with zeros at the end.

    If num_frames > num_frames_for_train:
        - Evenly pick frames from the original sequence using linear spacing 
          without introducing duplicates.
          
    Parameters:
    - pixel_array: A numpy array of shape (num_frames, image_height, image_width).
    - num_frames_for_train: The desired number of frames after adjustment.
    - image_size: The size of each frame (height == width).
    
    Returns:
    - A numpy array of shape (num_frames_for_train, image_size, image_size).
    """
    num_frames = pixel_array.shape[0]

    if num_frames == num_frames_for_train:
        # Already matches the desired number of frames
        return pixel_array

    if num_frames < num_frames_for_train:
        frames_to_add = num_frames_for_train - num_frames
        pre_frames, post_frames = divide_into_two_digits(frames_to_add)

        pre_mask = np.zeros([pre_frames, image_size, image_size], dtype=np.uint8)
        post_mask = np.zeros([post_frames, image_size, image_size], dtype=np.uint8)

        image = np.concatenate([pre_mask, pixel_array, post_mask], axis=0)
    else:
        # num_frames > num_frames_for_train
        # Evenly space indices across the full range [0, num_frames-1]
        indices = np.linspace(0, num_frames - 1, num_frames_for_train)
        indices = np.round(indices).astype(int)  # Convert to integers
        
        # Since M > N, spacing is >=1, so no duplicate indices should occur.
        # Still, let's assert just to be safe:
        # assert len(np.unique(indices)) == num_frames_for_train, "Duplicate indices found!"
        
        image = pixel_array[indices]

    return image
