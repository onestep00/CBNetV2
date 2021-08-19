_base_ = [
    "../_base_/models/htc_without_semantic_swin_fpn.py",
    # "../_base_/datasets/coco_instance.py",
    "../_base_/datasets/coco_detection.py",
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]

model = dict(
    backbone=dict(
        type="CBSwinTransformer",
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False,
    ),
    neck=dict(type="CBFPN", in_channels=[128, 256, 512, 1024]),
    roi_head=dict(
        bbox_head=[
            dict(
                type="ConvFCBBoxHead",
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                # class food
                num_classes=1,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2],
                ),
                reg_class_agnostic=True,
                reg_decoded_bbox=True,
                # single gpu
                norm_cfg=dict(type="BN", requires_grad=True),
                loss_cls=dict(
                    type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
                ),
                loss_bbox=dict(type="GIoULoss", loss_weight=10.0),
            ),
            dict(
                type="ConvFCBBoxHead",
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                # class food
                num_classes=1,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1],
                ),
                reg_class_agnostic=True,
                reg_decoded_bbox=True,
                # single gpu
                norm_cfg=dict(type="BN", requires_grad=True),
                loss_cls=dict(
                    type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
                ),
                loss_bbox=dict(type="GIoULoss", loss_weight=10.0),
            ),
            dict(
                type="ConvFCBBoxHead",
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                # class food
                num_classes=1,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067],
                ),
                reg_class_agnostic=True,
                reg_decoded_bbox=True,
                # single gpu
                norm_cfg=dict(type="BN", requires_grad=True),
                loss_cls=dict(
                    type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
                ),
                loss_bbox=dict(type="GIoULoss", loss_weight=10.0),
            ),
        ],
        # mask_head=[
        #     dict(
        #         type="HTCMaskHead",
        #         num_classes=2,
        #     ),
        #     dict(
        #         type="HTCMaskHead",
        #         num_classes=2,
        #     ),
        #     dict(
        #         type="HTCMaskHead",
        #         num_classes=2,
        #     ),
        # ],
        mask_roi_extractor=None,
        mask_head=None,
    ),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type="soft_nms"),
        )
    ),
)


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

# augmentation strategy originates from HTC
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=False, with_seg=False),
    dict(
        type="Resize",
        img_scale=[(1600, 400), (1600, 1400)],
        multiscale_mode="range",
        keep_ratio=True,
    ),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="SegRescale", scale_factor=1 / 8),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        # keys=["img", "gt_bboxes", "gt_labels", "gt_masks", "gt_semantic_seg"],
        keys=[
            "img",
            "gt_bboxes",
            "gt_labels",
        ],
    ),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1600, 1400),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]


dataset_type = "CocoDataset"
classes = ("음식",)
data_root = "/home/jovyan/data/filtered-food2"
anno_root = "/home/jovyan/workspace/ml_mg/json_data/"

samples_per_gpu = 1

data = dict(
    workers_per_gpu=16,
    samples_per_gpu=samples_per_gpu,
    train=dict(
        type=dataset_type,
        img_prefix=data_root,
        classes=classes,
        ann_file=anno_root + "train_new_split.json",
        # ann_file=anno_root + "datatest.json",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        img_prefix=data_root,
        classes=classes,
        ann_file=anno_root + "val_new_split.json",
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        img_prefix=data_root,
        classes=classes,
        ann_file=anno_root + "test_new.json",
        pipeline=test_pipeline,
    ),
)
optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=0.0001 * (samples_per_gpu / 2),
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)

lr_config = dict(step=[16, 19])
runner = dict(type="EpochBasedRunnerAmp", max_epochs=20)

# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)

log_config = dict(
    interval=1,
    hooks=[
        dict(type="TextLoggerHook", reset_flag=True),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="mmdetection",
                name="htc_cbv2_swin_base22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco",
            ),
        ),
    ],
)

evaluation = dict(  # The config to build the evaluation hook, refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/evaluation/eval_hooks.py#L7 for more details.
    interval=1, metric=["bbox"]  # Evaluation interval
)

workflow = [("train", 1), ("val", 1)]
# workflow = [("val", 1)]

resume_from = "/home/jovyan/workspace/ml_mg/cbnetev2/work_dirs/htc_cbv2_swin_base_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco/latest.pth"
# load_from = "/home/jovyan/workspace/ml_mg/cbnetev2/work_dirs/htc_cbv2_swin_base_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco/latest.pth"

# pretrained
# load_from = "/home/jovyan/workspace/ml_mg/cbnetev2/checkpoints/htc_cbv2_swin_base22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.pth"
