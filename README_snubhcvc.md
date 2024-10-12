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

## TODO
- [ ] normalization 잘 되고 있는지 확인해보기 (z-score normalization)
- [ ] `VideoResNet` pretrained weight 확인해보기
