## Installation
Refer to the `./Dockerfile`

## Dataset
- cag_coronary_classification

## Train
```
# CAG coronary classification
python tools/train.py configs/video_resnet/video_resnet_cag_coronary.py

# CAG frame selection
python tools/train.py configs/video_resnet/video_resnet_cag_frame.py
```

## Inference
Use scripts in `tools/inference`

```
# CAG frame selection
python tools/inference/cag_frame_selection.py work_dirs/video_resnet_cag_frame/video_resnet_cag_frame.py work_dirs/video_resnet_cag_frame/epoch_98.pth ./data/cag_frame_selection/test ./results/cag_frame_selection 
```


## TODO
- [ ] normalization 잘 되고 있는지 확인해보기 (z-score normalization)
- [ ] `VideoResNet` pretrained weight 확인해보기
