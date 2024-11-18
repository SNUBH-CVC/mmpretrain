from pathlib import Path
from typing import Optional

import numpy as np
from mmpretrain.registry import DATASETS
from .base_dataset import BaseDataset


@DATASETS.register_module()
class CAGVideo(BaseDataset):
    classes_str_to_int = {
        "class_valid": 1,
        "class_invalid": 0,
    }

    def __init__(self,
                 data_root: str = '',
                 split: str = 'train',
                 metainfo: Optional[dict] = None,
                 download: bool = True,
                 data_prefix: str = '',
                 test_mode: bool = False,
                 **kwargs):
        self.split = split
        super().__init__(
            # The CIFAR dataset doesn't need specify annotation file
            ann_file='',
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=dict(root=data_prefix),
            test_mode=test_mode,
            serialize_data=False,
            **kwargs)

    def get_data_info(self, idx):
        data = self.data_list[idx]
        img = np.load(data["img_path"])
        label = data["gt_label"]
        return {"img": img, "gt_label": label}

    def load_data_list(self):
        data_root = Path(self.data_root)
        split_dir_path = data_root / self.split

        data_list = []
        for k, v in self.classes_str_to_int.items():
            label = v
            img_dir_path = split_dir_path / k
            for img_path in img_dir_path.glob("*.npy"):
                data_list.append({"img_path": str(img_path), "gt_label": label})
        return data_list
