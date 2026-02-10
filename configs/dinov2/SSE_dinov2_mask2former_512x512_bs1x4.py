# dataset config
_base_ = [
    "../_base_/datasets/dg_V2PandRGB_512x512.py",
    #"../_base_/datasets/dg_PRGB2V_512x512.py",
    #"../_base_/datasets/dg_P2V_512x512.py",
    "../_base_/default_runtime.py",
    "../_base_/models/SSE_dinov2_mask2former.py",
]
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(
        type="RandomChoiceResize",
        scales=[int(512 * x * 0.1) for x in range(5, 21)],
        resize_type="ResizeShortestEdge",
        max_size=2048,
    ),
    dict(type="RandomCrop", crop_size={{_base_.crop_size}}, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
#train_dataloader = dict(batch_size=16, dataset=dict(pipeline=train_pipeline))
#train_dataloader = dict(batch_size=16)
'''
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(pipeline=train_pipeline),  # 添加 pipeline
            dict(pipeline=train_pipeline),  # 添加 pipeline
        ]
    )
)
'''
'''
#training dataset=Potsdan_IRRG
train_dataloader = dict(
    batch_size=16,  # 修改批次大小
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type={{_base_.Potsdam_IRRG_type}},
                data_root={{_base_.Potsdam_IRRG_root}},
                data_prefix=dict(
                    img_path="img_dir",
                    seg_map_path="ann_dir"
                ),
                pipeline=train_pipeline  # 确保使用训练流水线
            ),
            dict(
                type={{_base_.Potsdam_IRRG_type}},
                data_root={{_base_.Potsdam_IRRG_root}},
                data_prefix=dict(
                    img_path="stylized_img_dir",
                    seg_map_path="ann_dir"
                ),
                pipeline=train_pipeline  # 确保使用训练流水线
            )
        ]
    )
)
'''
'''
#training dataset=Potsdan_RGB
train_dataloader = dict(
    batch_size=16,  # 修改批次大小
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type={{_base_.Potsdam_RGB_type}},
                data_root={{_base_.Potsdam_RGB_root}},
                data_prefix=dict(
                    img_path="img_dir",
                    seg_map_path="ann_dir"
                ),
                pipeline=train_pipeline  # 确保使用训练流水线
            ),
            dict(
                type={{_base_.Potsdam_RGB_type}},
                data_root={{_base_.Potsdam_RGB_root}},
                data_prefix=dict(
                    img_path="stylized_img_dir",
                    seg_map_path="ann_dir"
                ),
                pipeline=train_pipeline  # 确保使用训练流水线
            )
        ]
    )
)
'''


#'''
#training dataset=vaihingen
train_dataloader = dict(
    batch_size=16,  # 修改批次大小
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type={{_base_.vaihingen_type}},
                data_root={{_base_.Vaihingen_root}},
                data_prefix=dict(
                    img_path="img_dir",
                    seg_map_path="ann_dir"
                ),
                pipeline=train_pipeline  # 确保使用训练流水线
            ),
            dict(
                type={{_base_.vaihingen_type}},
                data_root={{_base_.Vaihingen_root}},
                data_prefix=dict(
                    img_path="stylized_img_dir",
                    seg_map_path="ann_dir"
                ),
                pipeline=train_pipeline  # 确保使用训练流水线
            )
        ]
    )
)
#'''



# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    constructor="PEFTOptimWrapperConstructor",
    optimizer=dict(
        type="AdamW", lr=0.0001, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999)
    ),
    paramwise_cfg=dict(
        custom_keys={
            "norm": dict(decay_mult=0.0),
            "query_embed": embed_multi,
            "level_embed": embed_multi,
            "learnable_tokens": embed_multi,
            "reins.scale": embed_multi,
        },
        norm_decay_mult=0.0,
    ),
)
param_scheduler = [
    dict(type="PolyLR", eta_min=0, power=0.9, begin=0, end=40000, by_epoch=False)
]

# training schedule for 160k
train_cfg = dict(type="IterBasedTrainLoop", max_iters=40000, val_interval=4000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook", by_epoch=False, interval=10000, max_keep_ckpts=3
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)
