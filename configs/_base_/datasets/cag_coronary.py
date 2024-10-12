dataset_type = 'CAGCoronary'
data_preprocessor = dict(
    num_classes=2,
    # RGB format normalization parameters
    mean=[122.961],
    std=[50.847],
    # loaded images are already RGB format
    to_rgb=False)

prob = 0.5
train_pipeline = [
    dict(type='MonaiResized', keys=["img"], spatial_size=(-1, 256, 256)),
    dict(type='MonaiRandZoomd', keys=["img"], prob=prob, min_zoom=0.8, max_zoom=1.2),
    dict(type='MonaiRandRotated', keys=["img"], prob=prob, range_x=(-0.26, 0.26), padding_mode="zeros"),
    dict(type='MonaiRandGaussianNoised', keys=["img"], prob=prob),
    dict(type='MonaiRepeatChanneld', keys=["img"], repeats=3),
    dict(type='PackInputs'),
]

val_pipeline = [
    dict(type='MonaiResized', keys=["img"], spatial_size=(-1, 256, 256)),
    dict(type='MonaiRepeatChanneld', keys=["img"], repeats=3),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root='data/cag_coronary_classification',
        split='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root='data/cag_coronary_classification',
        split='val',
        pipeline=val_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_evaluator = dict(type="Accuracy", topk=(1, ))

test_dataloader = val_dataloader
test_evaluator = val_evaluator
