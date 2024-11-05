"""
python tools/inference/cag_coronary_classification.py work_dirs/video_resnet_cag_coronary/video_resnet_cag_coronary.py work_dirs/video_resnet_cag_coronary/epoch_59.pth /mnt/nas/snubhcvc/results/241101_cag_groupby_angle/ ./inference_result.csv
"""

import argparse
from pathlib import Path
import csv
import tqdm

from mmengine import Config
from mmengine.dataset import Compose
from mmpretrain import init_model
from mmpretrain.datasets.transforms import TRANSFORMS
from mmpretrain_utils.preprocess import load_and_process_dicom, adjust_frames  # Import necessary functions


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with custom MMPreTrain model")
    parser.add_argument("config_path", type=str, help="Path to the model config file")
    parser.add_argument("checkpoint_path", type=str, help="Path to the model checkpoint file")
    parser.add_argument("data_dir", type=str, help="Directory path containing DICOM files")
    parser.add_argument("output_path", type=str, default="inference_results.csv", help="Path to save the results")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the inference on, e.g., 'cuda:0' or 'cpu'")
    parser.add_argument("--image_size", type=int, default=512, help="Size to which the image will be resized")
    parser.add_argument("--num_frames_for_train", type=int, default=60, help="Number of frames for training")
    return parser.parse_args()


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

    # Prepare CSV file for output
    output_file = Path(args.output_path)
    processed_files = dict()

    # Read existing CSV to get already processed files
    if output_file.exists():
        with open(output_file, mode='r', newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader, None)  # Skip header
            for row in csv_reader:
                key = f"{row[0]}_{row[1]}_{row[2]}"
                processed_files[key] = True

    with open(output_file, mode='a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header if the file is empty
        if csvfile.tell() == 0:
            csv_writer.writerow(['patient_id', 'study_date', 'series_no', 'coronary_cls'])

        # Iterate over DICOM files in the directory
        data_dir = Path(args.data_dir)
        dcm_path_list = list(data_dir.glob("*/*/XA/*.dcm"))
        print(f"Processing {len(dcm_path_list)} DICOM files")
        for dcm_path in tqdm.tqdm(dcm_path_list, total=len(dcm_path_list)):
            patient_id = dcm_path.parents[2].name
            study_date = dcm_path.parents[1].name
            series_no = dcm_path.stem

            key = f"{patient_id}_{study_date}_{series_no}"
            if processed_files.get(key, False):
                continue

            # Load and process the DICOM file
            pixel_array = load_and_process_dicom(dcm_path, args.image_size)
            if pixel_array is None:
                continue

            # Adjust frames
            adjusted_frames = adjust_frames(pixel_array, args.num_frames_for_train, args.image_size)

            # Prepare inputs for the model
            inputs = {'img': adjusted_frames[None]}
            inputs = {'inputs': transforms(inputs)['inputs'].to(device)}
            data = model.data_preprocessor(inputs)
            img = data["inputs"].unsqueeze(0)

            # Run the model for prediction
            out = model(img, mode="predict")
            pred_label = out[0].pred_label.item()
            if pred_label == 0:
                pred_label = 'left'
            else:
                pred_label = 'right'

            # Write the result to the CSV file
            csv_writer.writerow([patient_id, study_date, series_no, pred_label])


if __name__ == "__main__":
    main()