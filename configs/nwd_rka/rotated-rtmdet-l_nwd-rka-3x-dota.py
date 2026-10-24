# NWD-RKA (Xu et al., ISPRS 2022) on rotated RTMDet-l: pure ranking assignment.
# Every GT takes its `topk` point priors ranked by NWD between the GT and a
# `point_prior_scale * stride` box at each prior as positives, so tiny GTs
# always get positives. Box quality for the QFL target / box-loss weight is the
# NWD between the decoded prediction and its GT.
_base_ = '../rotated_rtmdet/rotated_rtmdet_l-3x-dota.py'

# NWD normalisation constant C, roughly the average absolute object size
# sqrt(w * h) of the training set: compute it with
#   python tools/analysis_tools/nwd_constant.py <this config>
# 12.8 is the paper's AI-TOD value.
nwd_constant = 12.8

model = dict(
    train_cfg=dict(
        assigner=dict(
            _delete_=True,
            type='RankingAssigner',
            topk=9,
            neg_thr=0.3,
            point_prior_scale=4.0,
            iou_calculator=dict(type='RBboxNWD2D', constant=nwd_constant),
            quality_calculator=dict(type='RBboxNWD2D', constant=nwd_constant))))
