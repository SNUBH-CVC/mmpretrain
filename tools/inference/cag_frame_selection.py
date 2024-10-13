import argparse
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from mmengine import Config
from mmengine.dataset import Compose
from mmpretrain import init_model
from mmpretrain.datasets.transforms import TRANSFORMS


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with custom MMPreTrain model")
    parser.add_argument("config_path", type=str, help="Path to the model config file")
    parser.add_argument("checkpoint_path", type=str, help="Path to the model checkpoint file")
    parser.add_argument("image_dir", type=str, help="Directory containing input image files (NumPy .npy)")
    parser.add_argument("output_dir", type=str, help="Directory to save output files")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the inference on, e.g., 'cuda:0' or 'cpu'")
    return parser.parse_args()


def get_kth_largest_rank(arr, index):
    """
    Returns the rank (1-based) of the element at the specified index 
    in terms of largest values in the array.
    
    Parameters:
    arr (np.ndarray): The input array.
    index (int): The index of the element whose rank is to be found.
    
    Returns:
    int: The rank of the element at the given index.
    """
    # Get the indices that would sort the array in descending order
    sorted_indices = np.argsort(arr)[::-1]
    
    # Find the 1-based rank of the specified index
    k = np.where(sorted_indices == index)[0][0] + 1
    
    return k


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
    image_path_list = image_dir_path.glob("*.npy")

    for img_path in tqdm(image_path_list):
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


if __name__ == "__main__":
    main()

