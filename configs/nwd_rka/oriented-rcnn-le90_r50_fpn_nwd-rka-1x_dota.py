# NWD-RKA on Oriented R-CNN: the paper's two-stage setting. The RPN uses
# ranking assignment (top-2 anchors per GT by NWD; horizontal anchors vs
# rotated GTs), the R-CNN stage keeps rotated-IoU MaxIoUAssigner.
# RPN proposals are additionally filtered with NWD-NMS (from the earlier NWD
# paper, Triton kernel, no custom mmcv). The final detections keep
# nms_rotated: with a small constant NWD-NMS barely suppresses duplicates of
# large objects.
_base_ = '../oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py'

# See tools/analysis_tools/nwd_constant.py; 12.8 is the paper's AI-TOD value.
nwd_constant = 12.8
rpn_nms = dict(type='nwd_nms', iou_threshold=0.85, constant=nwd_constant)

model = dict(
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                _delete_=True,
                type='RankingAssigner',
                topk=2,
                neg_thr=0.3,
                gpu_assign_thr=512,
                iou_calculator=dict(type='RBboxNWD2D', constant=nwd_constant))),
        rpn_proposal=dict(nms=rpn_nms)),
    test_cfg=dict(rpn=dict(nms=rpn_nms)))
