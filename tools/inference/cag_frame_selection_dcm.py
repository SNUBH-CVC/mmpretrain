"""
python tools/inference/cag_frame_selection_dcm.py \
    work_dirs/video_resnet_cag_frame/video_resnet_cag_frame.py \
    work_dirs/video_resnet_cag_frame/epoch_48.pth \
    /mnt/nas/snubhcvc/raw/cag_ccta_1yr_all/data /mnt/nas/snubhcvc/raw/cpacs \
    --output_path results/241230_cag_frame_selection_dcm/cag_frame_selection_inference_results_3.csv
"""

import argparse
from pathlib import Path

from mmengine import Config
from mmengine.dataset import Compose
from mmpretrain import init_model
from mmpretrain.datasets.transforms import TRANSFORMS
from mmpretrain_utils.preprocess import load_and_process_dicom
from mmpretrain_utils.utils import get_mongodb_database, CSVManager


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with custom MMPreTrain model")
    parser.add_argument("config_path", type=str, help="Path to the model config file")
    parser.add_argument("checkpoint_path", type=str, help="Path to the model checkpoint file")
    parser.add_argument("data_dirs", nargs="+", type=str, help="Directory path containing DICOM files")
    parser.add_argument("--output_path", type=str, default="cag_frame_selection_inference_results.csv", help="Path to save the results")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device to run the inference on, e.g., 'cuda:0' or 'cpu'")
    return parser.parse_args()


def get_valid_xa_list_from_mongodb():
    db = get_mongodb_database()
    collection = db.get_collection("videos_prediction")
    return list(collection.find({"data.category.left_right": {"$in": [0, 1]}, "data.category.is_valid": 1}))


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

    output_file = Path(args.output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    valid_xa_list = get_valid_xa_list_from_mongodb()
    num_valid_xa = len(valid_xa_list)
    print(f"Processing {len(valid_xa_list)} DICOM files")

    with CSVManager(output_file, ['patient_id', 'study_date', 'series_no', 'frame_idx']) as csv_manager:
        for i, valid_xa in enumerate(valid_xa_list):
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

            key = f"{patient_id}_{study_date}_{int(series_no_str)}"
            if csv_manager.is_processed(key):
                continue

            # Load and process the DICOM file
            pixel_array = load_and_process_dicom(dcm_path, 512)
            if pixel_array is None:
                continue

            inputs = {'img': pixel_array[None]}
            inputs = transforms(inputs)
            idx_offset = inputs['idx_offset']
            inputs = {"inputs": inputs["inputs"].to(device)}
            data = model.data_preprocessor(inputs)
            img = data["inputs"].unsqueeze(0)

            # Run the model for prediction
            out = model(img, mode="predict")[0]

            # Visualize
            pred_label = out.pred_label.item()
            pred_label = pred_label + idx_offset
            print(f"{i}/{num_valid_xa} key: {key}")
            csv_manager.write_row([patient_id, study_date, series_no_str, pred_label])


if __name__ == "__main__":
    main()

