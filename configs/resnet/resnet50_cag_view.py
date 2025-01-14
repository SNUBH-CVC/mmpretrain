_base_ = [
    '../_base_/models/resnet50.py', '../_base_/datasets/cag_view.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]

model = dict(
    head=dict(
        num_classes=9
    )
)