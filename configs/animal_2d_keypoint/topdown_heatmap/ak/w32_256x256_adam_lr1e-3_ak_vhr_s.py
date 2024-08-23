_base_ = ['../../../_base_/default_runtime.py']

# runtime settings
auto_resume = True
gpus = [0]
output_dir = 'output/ak_P3_bird/vhr_birdpose_s'
log_dir = 'log/ak_P3_bird/vhr_birdpose_s'
cudnn_benchmark = True
cudnn_deterministic = False
cudnn_enabled = True

train_cfg = dict(max_epochs=150, val_interval=10)

# optimizer settings
optim_wrapper = dict(
    optimizer=dict(
        type='Adam',
        lr=0.001,
        weight_decay=0.0001,
        betas=(0.99, 0.0),
        nesterov=False
    )
)

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', 
        begin=0, 
        end=500, 
        start_factor=0.001,
        by_epoch=False  # warm-up
    ),
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True
    )
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(
    checkpoint=dict(save_best='PCK', rule='greater'),
    logger=dict(type='LoggerHook', interval=100)
)

# codec settings
codec = dict(
    type='MSRAHeatmap', 
    input_size=(256, 256), 
    heatmap_size=(64, 64), 
    sigma=2
)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True
    ),
    backbone=dict(
        type='VHRBirdPose',
        cfg=dict(
            MODEL=dict(
                NUM_JOINTS=32,
                EXTRA=dict(
                    PRETRAINED_LAYERS=[
                        'conv1', 'bn1', 'conv2', 'bn2', 'layer1', 
                        'transition1', 'stage2', 'transition2', 
                        'stage3', 'transition3', 'stage4'
                    ],
                    FINAL_CONV_KERNEL=1,
                    STAGE2=dict(
                        NUM_MODULES=1,
                        NUM_BRANCHES=2,
                        BLOCK='BASIC',
                        NUM_BLOCKS=[4, 4],
                        NUM_CHANNELS=[32, 64],
                        FUSE_METHOD='SUM'
                    ),
                    STAGE3=dict(
                        NUM_MODULES=4,
                        NUM_BRANCHES=3,
                        BLOCK='BASIC',
                        NUM_BLOCKS=[4, 4, 4],
                        NUM_CHANNELS=[32, 64, 128],
                        FUSE_METHOD='SUM'
                    ),
                    STAGE4=dict(
                        NUM_MODULES=3,
                        NUM_BRANCHES=4,
                        BLOCK='BASIC',
                        NUM_BLOCKS=[4, 4, 4, 4],
                        NUM_CHANNELS=[32, 64, 128, 256],
                        FUSE_METHOD='SUM'
                    ),
                    VIT=dict(
                        DEPTH=6,
                        NUM_HEADS=12
                    ),
                    FUSE_STREGY='add'
                ),
            ),
        ),
        in_channels=3,
        init_cfg=dict(type='Pretrained', checkpoint='models/pytorch/mpii/VHR-BirdPose-S.pth')
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=32,
        out_channels=23,
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec
    ),
    test_cfg=dict(
        flip_test=True,
        post_process=True,
        shift_heatmap=True
    )
)

# base dataset settings
dataset_type = 'AnimalKingdomDataset'
data_mode = 'topdown'
data_root = 'data/ak_P3_bird'
dataset_settings = dict(
    color_rgb=False,
    data_format='jpg',
    flip=True,
    num_joints_half_body=11,
    prob_half_body=-1.0,
    root=data_root,
    rot_factor=30,
    scale_factor=0.25
)

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=8,
    num_workers=14,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline,
    )
)
val_dataloader = dict(
    batch_size=8,
    num_workers=14,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/test.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=val_pipeline,
    )
)
test_dataloader = val_dataloader

# evaluators
val_evaluator = [
    dict(type='PCKAccuracy', thr=0.05),
    dict(type='AUC')
]
test_evaluator = val_evaluator

# debug settings
debug = dict(
    debug=True,
    save_batch_images_gt=True,
    save_batch_images_pred=True,
    save_heatmaps_gt=True,
    save_heatmaps_pred=True
)
