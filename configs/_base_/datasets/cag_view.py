dataset_type = 'CAGView'
data_preprocessor = dict(
    num_classes=9,
    # RGB format normalization parameters
    mean=[122.961],
    std=[50.847],
    # loaded images are already RGB format
    to_rgb=False)

prob = 0.5
train_pipeline = [
    dict(
        type='Albumentations',
        transforms=[
            dict(type='Affine', scale=[0.9, 1.0], rotate=[-5, 5], p=prob),
            dict(
                type='OneOf',
            transforms=[
                dict(type='Blur', blur_limit=3, p=1.0),
                dict(type='MedianBlur', blur_limit=3, p=1.0)
            ],
            p=prob),
        ],
    ),
    dict(type='PackInputs'),
]

val_pipeline = [
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root='data/cag_view_classification',
        split='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root='data/cag_view_classification',
        split='val',
        pipeline=val_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_evaluator = dict(type="Accuracy", topk=(1, ))

test_dataloader = val_dataloader
test_evaluator = val_evaluator

