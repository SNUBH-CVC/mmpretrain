import argparse
from pathlib import Path
import json
from collections import defaultdict
from shutil import copyfile
import random

from mmpretrain_utils.preprocess import split_dataset_by_patient

PRETRAIN_VAL_IMAGE_DIR = Path("/mnt/nas/snubhcvc/project/cagfm/pretrain/datasets/images/val")
VIEW_PREDICTION_DATA_DIR = Path("/mnt/nas/snubhcvc/project/cagfm/downstream_tasks/view_prediction/datasets")
VIEW_INFO_JSON_PATH = VIEW_PREDICTION_DATA_DIR / "val_view_info.json"
random.seed(42)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert CAG angle prediction dataset")
    parser.add_argument("output_dir", type=str, help="Path to the input directory")
    parser.add_argument("--test_size", type=float, default=0.1, help="Fraction of the data to use for testing")
    parser.add_argument("--val_size", type=float, default=0.1, help="Fraction of the data to use for validation")
    parser.add_argument("--num_samples", type=int, nargs="+", default=[200, 200, 200], help="Number of samples to use for each split")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(VIEW_INFO_JSON_PATH, "r") as f:
        view_info_data = json.load(f)

    image_path_list = PRETRAIN_VAL_IMAGE_DIR.glob("*.png")
    all_data = []
    for img_path in image_path_list:
        patient_id, study_date, series_no_str, _ = img_path.stem.split("_")
        key = f"{patient_id}_{study_date}_{series_no_str}"
        view = view_info_data[key]["view"]
        left_right = view_info_data[key]["left_right"]
        all_data.append({"img_path": img_path, "view": view, "left_right": left_right, "patient_id": patient_id})

    train_data, val_data, test_data = split_dataset_by_patient(all_data, args.test_size, args.val_size)

    for mode, data_list, num_samples in zip(["train", "val", "test"], [train_data, val_data, test_data], args.num_samples):
        print(f"Processing {mode} dataset")
        # left_right 기준까지 추가 고려했는데 일부 view에서는 개수가 부족한 경우가 있어서 비율 반영을 위해 랜덤으로 샘플링
        random.shuffle(data_list)
        view_count = defaultdict(int)
        for view_info_data in data_list:
            img_path = view_info_data["img_path"]
            view = view_info_data["view"]
            view_dir = output_dir / mode / view
            view_dir.mkdir(parents=True, exist_ok=True)
            if view_count[view] >= num_samples:
                break
            save_path = view_dir / img_path.name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            copyfile(img_path, save_path)
            view_count[view] += 1


if __name__ == "__main__":
    main()
