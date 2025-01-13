"""
python tools/inference/cag_video_classification.py \
    work_dirs/video_resnet_cag_video/video_resnet_cag_video.py \
    work_dirs/video_resnet_cag_video/epoch_65.pth \
    /mnt/nas/snubhcvc/raw/cag_ccta_1yr_all/data /mnt/nas/snubhcvc/raw/cpacs \
    --output_path cag_video_classification_inference_results_1.csv
"""

import argparse
from pathlib import Path
import tqdm

from mmengine import Config
from mmengine.dataset import Compose
from mmpretrain import init_model
from mmpretrain.datasets.transforms import TRANSFORMS
from mmpretrain_utils.utils import get_mongodb_database, CSVManager
from mmpretrain_utils.preprocess import load_and_process_dicom, resample_frames_evenly  # Import necessary functions


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with custom MMPreTrain model")
    parser.add_argument("config_path", type=str, help="Path to the model config file")
    parser.add_argument("checkpoint_path", type=str, help="Path to the model checkpoint file")
    parser.add_argument("data_dirs", nargs="+", type=str, help="Directory path containing DICOM files")
    parser.add_argument("--output_path", type=str, default="cag_video_classification_inference_results.csv", help="Path to save the results")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device to run the inference on, e.g., 'cuda:0' or 'cpu'")
    parser.add_argument("--image_size", type=int, default=512, help="Size to which the image will be resized")
    parser.add_argument("--num_frames_for_train", type=int, default=60, help="Number of frames for training")
    return parser.parse_args()


def get_valid_xa_list_from_mongodb():
    db = get_mongodb_database()
    collection = db.get_collection("videos_prediction")
    return list(collection.find({"data.category.is_valid": 1}))


def main():
    # Set device
    args = parse_args()
    device = args.device

    # Load configuration and initialize model
    cfg = Config.fromfile(args.config_path)
    model = init_model(cfg, args.checkpoint_path).to(device)
    model.eval()

    # Initialize data transforms
    transforms = []
    for transform in cfg.test_dataloader.dataset.pipeline:
        transforms.append(TRANSFORMS.build(transform))
    transforms = Compose(transforms)

    # Iterate over DICOM files in the directory
    valid_xa_list = get_valid_xa_list_from_mongodb()
    num_valid_xa = len(valid_xa_list)
    print(f"Processing {num_valid_xa} DICOM files")

    with CSVManager(args.output_path, ['patient_id', 'study_date', 'series_no', 'video_cls']) as csv_manager:
        for valid_xa in tqdm.tqdm(valid_xa_list, total=len(valid_xa_list)):
            filename = valid_xa["filename"]
            patient_id, study_date, _, basename = filename.split("/")
            series_no_str = basename.split(".")[0]
            for data_dir in args.data_dirs: 
                dcm_path = Path(data_dir) / filename
                if dcm_path.exists():
                    break
            if not dcm_path.exists():
                print(f"DICOM file not found: {dcm_path}")
                continue

            key = f"{patient_id}_{study_date}_{series_no_str}"
            if csv_manager.is_processed(key):
                continue

            # Load and process the DICOM file
            pixel_array = load_and_process_dicom(dcm_path, args.image_size)
            if pixel_array is None:
                continue

            # Adjust frames
            adjusted_frames = resample_frames_evenly(pixel_array, args.num_frames_for_train, args.image_size)

            # Prepare inputs for the model
            inputs = {'img': adjusted_frames[None]}
            inputs = {'inputs': transforms(inputs)['inputs'].to(device)}
            data = model.data_preprocessor(inputs)
            img = data["inputs"].unsqueeze(0)

            # Run the model for prediction
            out = model(img, mode="predict")
            pred_label = out[0].pred_label.item()
            if pred_label == 0:
                pred_label = 'invalid'
            else:
                pred_label = 'valid'

            # Write the result to the CSV file
            csv_manager.write_row([patient_id, study_date, series_no_str, pred_label])


if __name__ == "__main__":
    main()
