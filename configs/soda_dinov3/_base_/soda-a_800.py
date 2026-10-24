# SODA-A, converted to DOTA format and split into 800x800 patches (stride 650):
#   tools/data/soda_a/README.md
dataset_type = 'DOTADataset'
data_root = '/mnt/4TB/Datasets/soda-a/split_800/'
backend_args = None
metainfo = dict(
    classes=('airplane', 'helicopter', 'small-vehicle', 'large-vehicle',
             'ship', 'container', 'storage-tank', 'swimming-pool',
             'windmill'),
    palette=[(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
             (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
             (255, 193, 193)])

# Patches are already 800x800: no resizing, so small objects keep their
# native resolution.
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs')
]
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    # No-op for the 800x800 patches; sets the scale_factor the head expects.
    dict(type='mmdet.Resize', scale=(800, 800), keep_ratio=True),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='train/annfiles/',
        data_prefix=dict(img_path='train/images/'),
        img_suffix='jpg',
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))
# val_mini: every 8th val scene (tools/data/soda_a/make_val_mini.py), used for
# the learning curves during training.
val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='val_mini/annfiles/',
        data_prefix=dict(img_path='val_mini/images/'),
        img_suffix='jpg',
        test_mode=True,
        pipeline=val_pipeline))
# Final numbers: the full (annotated) SODA-A test split.
test_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='test/annfiles/',
        data_prefix=dict(img_path='test/images/'),
        img_suffix='jpg',
        test_mode=True,
        pipeline=val_pipeline))

val_evaluator = dict(type='SODAMetric')
test_evaluator = val_evaluator
