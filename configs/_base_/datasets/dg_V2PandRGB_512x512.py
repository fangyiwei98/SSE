_base_ = [
    "./potsdam_IRRG_512x512.py",
    "./vaihingen_IRRG_512x512.py",
    "./potsdam_RGB_512x512.py",
]
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            {{_base_.train_Vaihingen}},
            {{_base_.train_Vaihingen_stylized}},
        ]
    )
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="ConcatDataset",
        datasets=[
            {{_base_.val_Potsdam_IRRG}},
            {{_base_.val_Potsdam_RGB}},
        ],
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="ConcatDataset",
        datasets=[
            # {{_base_.val_Potsdam_RGB}},
            {{_base_.val_Potsdam_IRRG}},
        ],
    ),
)
val_evaluator = dict(
    type="DGIoUMetric",
    iou_metrics=["mIoU"],
    dataset_keys=["Potsdam_IRRG/", "Potsdam_RGB/"],
    mean_used_keys=["Potsdam_IRRG/", "Potsdam_RGB/"],
)

test_evaluator = dict(
    type="IoUMetric",
    iou_metrics=["mIoU"],
    format_only=False,
    output_dir="./work_dirs/try",
)