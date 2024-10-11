import argparse
import numpy as np
import torch
from mmengine import Config
from mmengine.dataset import Compose
from mmpretrain import init_model
from mmpretrain.datasets.transforms import TRANSFORMS


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with custom MMPreTrain model")
    parser.add_argument("--config-path", type=str, required=True, help="Path to the model config file")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to the model checkpoint file")
    parser.add_argument("--image-path", type=str, required=True, help="Path to the input image file (NumPy .npy)")
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
    img = np.load(args.image_path)
    img = torch.tensor(img).to(device)
    inputs = {'img': img}
    inputs = {'inputs': transforms(inputs)['inputs']}
    data = model.data_preprocessor(inputs)
    img = data["inputs"].unsqueeze(0)

    # Run the model for prediction
    out = model(img, mode="predict")
    print(out)


if __name__ == "__main__":
    main()
