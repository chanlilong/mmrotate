# CPU-friendly smoke test of NWD-RKA on synthetic data:
#   python tools/data/dota/make_proxy_dota.py --out data/proxy_dota
#   python tools/train.py configs/nwd_rka/rotated-rtmdet-tiny_nwd-rka-proxy.py
_base_ = '../rotated_rtmdet/rotated_rtmdet_tiny-3x-dota.py'

data_root = 'data/proxy_dota/'
nwd_constant = 12.8

model = dict(
    backbone=dict(init_cfg=None, norm_cfg=dict(type='BN')),
    neck=dict(norm_cfg=dict(type='BN')),
    bbox_head=dict(norm_cfg=dict(type='BN')),
    train_cfg=dict(
        assigner=dict(
            _delete_=True,
            type='RankingAssigner',
            topk=9,
            point_prior_scale=4.0,
            iou_calculator=dict(type='RBboxNWD2D', constant=nwd_constant),
            quality_calculator=dict(type='RBboxNWD2D', constant=nwd_constant))))

train_dataloader = dict(
    batch_size=2, num_workers=2, dataset=dict(data_root=data_root))
val_dataloader = dict(num_workers=2, dataset=dict(data_root=data_root))
test_dataloader = val_dataloader

train_cfg = dict(max_epochs=2, val_interval=2)
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-3, by_epoch=False, begin=0, end=20)
]
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=5),
    checkpoint=dict(type='CheckpointHook', interval=2, max_keep_ckpts=1))
