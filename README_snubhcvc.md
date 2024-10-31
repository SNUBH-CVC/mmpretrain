## Installation
Refer to the `./Dockerfile`
```
wandb login
```

## Dataset
- cag_coronary_classification

## Train
```
# CAG coronary classification
CUDA_VISIBLE_DEVICES=1 PORT=29500 tools/dist_train.sh configs/video_resnet/video_resnet_cag_coronary.py 1

# CAG frame selection
CUDA_VISIBLE_DEVICES=1 PORT=29500 tools/dist_train.sh configs/video_resnet/video_resnet_cag_frame.py 1
```

## Inference
Use scripts in `tools/inference`
```
# CAG frame selection
python tools/inference/cag_frame_selection.py work_dirs/video_resnet_cag_frame/video_resnet_cag_frame.py work_dirs/video_resnet_cag_frame/epoch_98.pth ./data/cag_frame_selection/test ./results/cag_frame_selection 
```

## Hooks
metric 기준 save_best 하고 싶은 경우 
https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.hooks.CheckpointHook.html

## TODO
- [x] normalization 잘 되고 있는지 확인해보기 (z-score normalization)
