# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VideoResNet'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=4096,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),  # soft-cross_entropy
        topk=(1, 5),
    ))
