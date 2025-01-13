"""
python tools/inference/cag_frame_selection.py \
    work_dirs/video_resnet_cag_frame/video_resnet_cag_frame.py \
    work_dirs/video_resnet_cag_frame/epoch_48.pth \
    ./data/cag_frame_selection/test \
    ./results/241230_cag_frame_selection
"""

import argparse
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from mmengine import Config
from mmengine.dataset import Compose
from mmpretrain import init_model
from mmpretrain.datasets.transforms import TRANSFORMS
from mmpretrain_utils.utils import get_kth_largest_rank


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with custom MMPreTrain model")
    parser.add_argument("config_path", type=str, help="Path to the model config file")
    parser.add_argument("checkpoint_path", type=str, help="Path to the model checkpoint file")
    parser.add_argument("image_dir", type=str, help="Directory containing input image files (NumPy .npy)")
    parser.add_argument("output_dir", type=str, help="Directory to save output files")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the inference on, e.g., 'cuda:0' or 'cpu'")
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

    # Load and preprocess the image
    image_dir_path = Path(args.image_dir)
    output_dir_path = Path(args.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    image_path_list = list(image_dir_path.glob("*.npy"))

    diff_histogram_data = []
    top_k_histogram_data = []

    for img_path in tqdm(image_path_list, total=len(image_path_list)):
        gt_label = int(img_path.stem.split("_")[-1])
        img = np.load(img_path)
        inputs = {'img': img, "gt_label": gt_label}
        inputs = transforms(inputs)
        gt_label = inputs['data_samples'].gt_label
        inputs = {"inputs": inputs["inputs"].to(device)}
        data = model.data_preprocessor(inputs)
        img = data["inputs"].unsqueeze(0)

        # Run the model for prediction
        out = model(img, mode="predict")[0]

        # Visualize
        gt_label = gt_label.item()
        pred_label = out.pred_label.item()
        pred_scores = out.pred_score.cpu().detach().numpy()
        gt_top_k = get_kth_largest_rank(pred_scores, gt_label)
        pred_score = pred_scores[pred_label]
        gt_score = pred_scores[gt_label]
        gt_pred_diff = abs(gt_label - pred_label)
        diff_histogram_data.append(gt_pred_diff)
        top_k_histogram_data.append(gt_top_k)

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        pred_frame_idx = out.pred_label[0]
        imgs = inputs['inputs'][0].cpu().numpy()
        pred_img = imgs[pred_frame_idx]
        gt_img = imgs[gt_label]
        axes[0].imshow(gt_img, cmap='gray')
        axes[0].set_title(f"[ground_truth] index: {gt_label}, score: {gt_score:03f}, top_k: {gt_top_k}")
        axes[1].imshow(pred_img, cmap='gray')
        axes[1].set_title(f"[prediction] index: {pred_label}, score: {pred_score:03f}")
        output_path = output_dir_path / f"{img_path.stem}.png"
        fig.savefig(output_path)
        plt.close(fig)

    diff_histogram_data = np.array(diff_histogram_data)
    plt.hist(diff_histogram_data, bins=np.arange(0, 61))
    plt.savefig(output_dir_path / "diff_histogram.png")
    plt.close()

    top_k_histogram_data = np.array(top_k_histogram_data)
    plt.hist(top_k_histogram_data, bins=np.arange(0, 61))
    plt.savefig(output_dir_path / "top_k_histogram.png")
    plt.close()


if __name__ == "__main__":
    main()

