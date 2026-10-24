# Hybrid NWD-RKA + dynamic soft-label assignment on rotated RTMDet-l.
# Stock DSL only considers priors whose centre lies inside a GT; here each GT
# also contributes its `rka_topk` priors ranked by NWD as candidates, and the
# DSL cost / dynamic-k / soft labels use NWD instead of rotated IoU.
_base_ = '../rotated_rtmdet/rotated_rtmdet_l-3x-dota.py'

# See rotated-rtmdet-l_nwd-rka-3x-dota.py / tools/analysis_tools/nwd_constant.py
nwd_constant = 12.8

model = dict(
    train_cfg=dict(
        assigner=dict(
            _delete_=True,
            type='RankingDynamicSoftLabelAssigner',
            topk=13,
            rka_topk=3,
            rka_prior_scale=4.0,
            rka_calculator=dict(type='RBboxNWD2D', constant=nwd_constant),
            iou_calculator=dict(type='RBboxNWD2D', constant=nwd_constant))))
