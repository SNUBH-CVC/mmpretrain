_base_ = [
    '../_base_/models/video_resnet.py', '../_base_/datasets/cag_video.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]

model = dict(
    head=dict(
        num_classes=2,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),  
    )
)
