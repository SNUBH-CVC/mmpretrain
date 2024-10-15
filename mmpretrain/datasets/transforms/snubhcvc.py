import numpy as np
from monai.transforms import Transform

from mmpretrain.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadCagSingleFrameSelectionData(Transform):
    def __init__(self, num_sample_frames=60, buffer=5, random=True):
        assert 2 * buffer < num_sample_frames
        self.num_sample_frames = num_sample_frames
        self.buffer = buffer
        self.random = random

    def __call__(self, data):
        assert "img" in data.keys() and "gt_label" in data.keys()

        frames = data["img"]
        C, T, H, W = frames.shape
        frames_dtype = frames.dtype
        num_frames = T
        frame_idx = data["gt_label"]  # e.g., 30

        if self.random:
            # Random selection of frames around the frame_idx
            new_s_idx_range = list(
                range(
                    frame_idx + self.buffer - self.num_sample_frames + 1,
                    frame_idx - self.buffer + 1,
                )
            )
            new_s_idx = np.random.choice(new_s_idx_range)
            new_e_idx = new_s_idx + self.num_sample_frames - 1

            new_frames = frames[:, max(0, new_s_idx): min(num_frames, new_e_idx + 1)]
            if new_s_idx < 0:
                new_frames = np.concatenate(
                    [
                        np.zeros([C, -new_s_idx, H, W], dtype=frames_dtype),
                        new_frames,
                    ],
                    axis=1,
                )
            if num_frames < new_e_idx + 1:
                new_frames = np.concatenate(
                    [
                        new_frames,
                        np.zeros([C, new_e_idx + 1 - num_frames, H, W], dtype=frames_dtype),
                    ],
                    axis=1,
                )
            new_frame_idx = int(frame_idx - new_s_idx)

        else:
            # Non-random: either zero-pad or slice middle range
            if num_frames >= self.num_sample_frames:
                # Slice middle range including gt_label index
                s_idx = max(0, frame_idx - self.num_sample_frames // 2)
                e_idx = min(num_frames - 1, s_idx + self.num_sample_frames - 1)
                s_idx = e_idx - self.num_sample_frames + 1 # Ensure exact size
                new_frames = frames[:, s_idx: e_idx + 1]
                new_frame_idx = frame_idx - s_idx
            else:
                pad_left = (self.num_sample_frames - num_frames) // 2
                pad_right = self.num_sample_frames - num_frames - pad_left
                new_frames = np.concatenate(
                    [
                        np.zeros([C, pad_left, H, W], dtype=frames_dtype),
                        frames,
                        np.zeros([C, pad_right, H, W], dtype=frames_dtype),
                    ],
                    axis=1,
                )
                new_frame_idx = frame_idx + pad_left

        assert new_frames.shape[1] == self.num_sample_frames

        data["img"] = new_frames
        data["gt_label"] = new_frame_idx

        return data


def main():
    # test cases
    transform = LoadCagSingleFrameSelectionData(random=False)

    # insufficient case
    img = np.zeros([1, 30, 32, 32])
    gt_label = 5
    out = transform({'img': img, 'gt_label': gt_label})
    assert out['gt_label'] == 20

    # sufficient case
    img = np.zeros([1, 100, 32, 32])
    gt_label = 5
    out = transform({'img': img, 'gt_label': gt_label})
    assert out['gt_label'] == 5

    gt_label = 50
    out = transform({'img': img, 'gt_label': gt_label})
    assert out['gt_label'] == 30

    # exact case
    img = np.zeros([1, 60, 32, 32])
    gt_label = 5
    out = transform({'img': img, 'gt_label': gt_label})
    assert out['gt_label'] ==5
