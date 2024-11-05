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


def adjust_frames(pixel_array, num_frames_for_train, image_size):
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