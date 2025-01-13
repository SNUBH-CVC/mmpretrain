## Installation
Refer to the `./Dockerfile`
```
wandb login
# Enter API key: 5d8f48d343dbe1917bfabf116499fcd50fc8f3ee
```

## Dataset
- cag_coronary_classification

## Train
```
# CAG video classification (valid/invalid)
CUDA_VISIBLE_DEVICES=0 PORT=29500 tools/dist_train.sh configs/video_resnet/video_resnet_cag_video.py 1

# CAG coronary classification
CUDA_VISIBLE_DEVICES=1 PORT=29500 tools/dist_train.sh configs/video_resnet/video_resnet_cag_coronary.py 1

# CAG frame selection
CUDA_VISIBLE_DEVICES=1 PORT=29500 tools/dist_train.sh configs/video_resnet/video_resnet_cag_frame.py 1
```

## Inference
Use scripts in `tools/inference`
```
# CAG video classification (valid/invalid)
python tools/inference/cag_video_classification.py work_dirs/video_resnet_cag_video/video_resnet_cag_video.py work_dirs/video_resnet_cag_video/epoch_19.pth ./data/cag_video_classification/test ./results/cag_video_classification

# CAG frame selection
python tools/inference/cag_frame_selection.py work_dirs/video_resnet_cag_frame/video_resnet_cag_frame.py work_dirs/video_resnet_cag_frame/epoch_98.pth ./data/cag_frame_selection/test ./results/cag_frame_selection 

# CAG coronary classification
python tools/inference/cag_coronary_classification.py work_dirs/video_resnet_cag_coronary/video_resnet_cag_coronary.py work_dirs/video_resnet_cag_coronary/epoch_59.pth /mnt/nas/snubhcvc/raw/cpacs ./coronary_cls_inference_result.csv
```

## Hooks
metric 기준 save_best 하고 싶은 경우 
https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.hooks.CheckpointHook.html
