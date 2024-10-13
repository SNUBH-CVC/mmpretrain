dataset_type = 'CAGFrame'
num_classes = 60
data_preprocessor = dict(
    num_classes=num_classes,
    # RGB format normalization parameters
    mean=[122.961],
    std=[50.847],
    # loaded images are already RGB format
    to_rgb=False)

prob = 0.5
train_pipeline = [
    dict(type='LoadCagSingleFrameSelectionData', num_sample_frames=num_classes, buffer=5),
    dict(type='MonaiResized', keys=["img"], spatial_size=(-1, 256, 256)),
    dict(type='MonaiRandRotated', keys=["img"], prob=prob, range_x=(-0.26, 0.26), padding_mode="zeros"),
    dict(type='MonaiRepeatChanneld', keys=["img"], repeats=3),
    dict(type='PackInputs'),
]

val_pipeline = [
    dict(type='LoadCagSingleFrameSelectionData', num_sample_frames=num_classes, buffer=5, random=False),
    dict(type='MonaiResized', keys=["img"], spatial_size=(-1, 256, 256)),
    dict(type='MonaiRepeatChanneld', keys=["img"], repeats=3),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root='data/cag_frame_selection',
        split='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root='data/cag_frame_selection',
        split='val',
        pipeline=val_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_evaluator = dict(type="Accuracy", topk=(1, 5))  

test_dataloader = val_dataloader
test_evaluator = val_evaluator
