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
                # class multi
                num_classes=163,
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
                # class multi
                num_classes=163,
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
                # class multi
                num_classes=163,
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
# classes = ("음식",)
classes = (
    "기타",
    "가래떡",
    "어묵볶음",
    "쌀밥",
    "배추김치",
    "라면류",
    "닭찜",
    "육류튀김",
    "김치찌개",
    "케이크류",
    "잡곡밥",
    "두부",
    "제육볶음",
    "열무김치",
    "보리밥",
    "기타빵류",
    "돼지등갈비찜",
    "치킨류",
    "중식면류",
    "달걀찜",
    "조미김",
    "감자볶음",
    "미역국",
    "김밥",
    "국수류",
    "기타반찬",
    "김치찜",
    "기타김치",
    "스파게티류",
    "기타떡",
    "토마토",
    "치즈",
    "기타구이",
    "등심스테이크",
    "볶음밥류",
    "참외",
    "버섯볶음",
    "샐러드",
    "연근조림",
    "죽류",
    "기타소스/기타장류",
    "돼지고기 수육",
    "덮밥",
    "젓갈",
    "돈까스",
    "시금치나물",
    "포도",
    "앙금빵류",
    "상추",
    "들깻잎",
    "육류전",
    "달걀프라이",
    "채소류튀김",
    "코다리찜",
    "기타불고기",
    "돼지고기구이",
    "버거류",
    "된장국",
    "채소",
    "떡볶이",
    "낙지볶음",
    "비빔밥",
    "사과",
    "피자류",
    "숙주나물",
    "애호박볶음",
    "멸치볶음",
    "생선구이",
    "깻잎장아찌",
    "콩조림",
    "카레(커리)",
    "돼지고기채소볶음",
    "바나나",
    "파프리카",
    "고사리나물",
    "미역줄기볶음",
    "콩나물국",
    "소불고기",
    "떠먹는요구르트",
    "햄",
    "소고기구이",
    "버섯구이",
    "오이",
    "된장찌개",
    "무생채",
    "어패류튀김",
    "키위",
    "리조또",
    "오징어볶음",
    "샌드위치류",
    "만두류",
    "과자",
    "채소류전",
    "시리얼",
    "순두부찌개",
    "귤",
    "딸기",
    "기타스테이크",
    "잡채",
    "오리불고기",
    "취나물",
    "가지볶음",
    "삶은달걀",
    "크림빵류",
    "부침류",
    "어패류전",
    "한과류",
    "소갈비찜",
    "메추리알 장조림",
    "안심스테이크",
    "단호박찜",
    "식빵류",
    "시래기나물",
    "아귀찜",
    "김치볶음",
    "우엉조림",
    "감",
    "돼지불고기",
    "고기장조림",
    "두부조림",
    "오징어채볶음",
    "즉석밥",
    "오삼불고기",
    "현미밥",
    "파김치",
    "페이스트리(파이)류",
    "총각김치",
    "닭가슴살",
    "해물찜",
    "도넛류",
    "마시는요구르트",
    "돼지갈비찜",
    "함박스테이크",
    "오징어찜",
    "오이나물",
    "컵/액체류용기",
    "삶은브로콜리",
    "청국장찌개",
    "그라탕",
    "적류",
    "소고기채소볶음",
    "조기찜",
    "제품사진",
    "기타해조류",
    "기타장아찌/절임류",
    "기타나물/숙채/생채/무침류",
    "기타조림",
    "기타국/찌개/탕",
    "기타튀김",
    "기타볶음",
    "기타난류",
    "기타찜",
    "기타면류",
    "견과류",
    "기타채소류",
    "기타과실류",
    "크래커",
    "기타전/적/부침류",
    "기타밥류",
    "기타죽/스프류",
    "도토리묵무침",
    "튀김빵류",
    "기타과자류",
)
data_root = "/home/jovyan/data/filtered-food3"
anno_root = "/home/jovyan/workspace/ml_mg/json_data/"

samples_per_gpu = 1

data = dict(
    workers_per_gpu=16,
    samples_per_gpu=samples_per_gpu,
    train=dict(
        type=dataset_type,
        img_prefix=data_root,
        classes=classes,
        ann_file=anno_root + "163train.json",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        img_prefix=data_root,
        classes=classes,
        ann_file=anno_root + "163val.json",
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        img_prefix=data_root,
        classes=classes,
        ann_file=anno_root + "163test.json",
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
runner = dict(type="EpochBasedRunnerAmp", max_epochs=40)

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
                name="163_class_htc_cbv2_swin_base22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco",
            ),
        ),
    ],
)

evaluation = dict(  # The config to build the evaluation hook, refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/evaluation/eval_hooks.py#L7 for more details.
    interval=1, metric=["bbox"]  # Evaluation interval
)

workflow = [("train", 5), ("val", 1)]
# workflow = [("val", 1)]

resume_from = "/home/jovyan/workspace/ml_mg/cbnetev2/work_dirs/163_htc_cbv2_swin_base_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco/latest.pth"
# load_from = "/home/jovyan/workspace/ml_mg/cbnetev2/work_dirs/htc_cbv2_swin_base_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco/latest.pth"

# pretrained
# load_from = "/home/jovyan/workspace/ml_mg/cbnetev2/checkpoints/htc_cbv2_swin_base22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.pth"
