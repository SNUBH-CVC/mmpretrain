_base_ = [
    '../_base_/models/video_resnet.py', '../_base_/datasets/cag_frame.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]

model = dict(
    head=dict(
        num_classes=60,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
    )
)

visualizer = dict(
    type='UniversalVisualizer', vis_backends=[dict(type='WandbVisBackend')])