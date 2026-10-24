# NWD-RKA on rotated RetinaNet: the paper's single-stage setting (top-3 anchors
# per GT ranked by NWD; best NWD < 0.3 -> negative; the rest ignored).
_base_ = '../rotated_retinanet/rotated-retinanet-rbox-le90_r50_fpn_1x_dota.py'

# See tools/analysis_tools/nwd_constant.py; 12.8 is the paper's AI-TOD value.
nwd_constant = 12.8

model = dict(
    train_cfg=dict(
        assigner=dict(
            _delete_=True,
            type='RankingAssigner',
            topk=3,
            neg_thr=0.3,
            gpu_assign_thr=512,
            iou_calculator=dict(type='RBboxNWD2D', constant=nwd_constant))))
