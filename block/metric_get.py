import torch


def iou(pred, true):  # 输入为(batch,(x_min,y_min,w,h))
    x1 = torch.max(pred[:, 0], true[:, 0])
    y1 = torch.max(pred[:, 1], true[:, 1])
    x2 = torch.min(pred[:, 0] + pred[:, 2], true[:, 0] + true[:, 2])
    y2 = torch.min(pred[:, 1] + pred[:, 3], true[:, 1] + true[:, 3])
    zeros = torch.zeros(len(pred)).to(pred.device)
    intersection = torch.max(x2 - x1, zeros) * torch.max(y2 - y1, zeros)
    union = (pred[:, 2]) * (pred[:, 3]) + (true[:, 2]) * (true[:, 3]) - intersection
    return intersection / union


def tp_tn_fp_fn(pred, true, judge, confidence_threshold, iou_threshold):  # 对网络输出(单个/批量)求非极大值抑制前的指标
    tp = 0
    tn = 0
    tp_fn = 0
    tn_fp = 0
    for i in range(len(pred)):
        if True in judge[i]:  # 有需要预测的位置
            pred_judge = pred[i][judge[i]]
            true_judge = true[i][judge[i]]
            pred_frame = pred_judge[..., 0:4]
            true_frame = true_judge[..., 0:4]
            judge_opposite = ~judge[i]  # True和False取反
            pred_confidence = pred_judge[..., 4]  # 需要预测的位置
            pred_confidence_opposite = pred[i][judge_opposite][..., 4]  # 不需要预测的位置
            pred_class = torch.argmax(pred_judge[..., 5:], dim=1)
            true_class = torch.argmax(pred_judge[..., 5:], dim=1)
            tp_judge = torch.where((pred_confidence >= confidence_threshold) & (pred_class == true_class) &
                                   (iou(pred_frame, true_frame) > iou_threshold), True, False)
            judge_tn = torch.where(pred_confidence_opposite < confidence_threshold, True, False)
            tp_fn += len(pred_confidence)
            tp += len(pred_confidence[tp_judge])
            tn_fp += len(pred_confidence_opposite)
            tn += len(pred_confidence_opposite[judge_tn])
        else:  # 所有位置都不需要预测
            pred_confidence_opposite = pred[i][..., 4]  # 不需要预测的位置
            judge_tn = torch.where(pred_confidence_opposite < confidence_threshold, True, False)
            tn_fp += len(pred_confidence_opposite)
            tn += len(pred_confidence_opposite[judge_tn])
    # 计算指标
    fp = tn_fp - tn
    fn = tp_fn - tp
    return tp, tn, fp, fn


def nms_tp_fn_fp(pred, true, iou_threshold):  # 输入为(batch,(x_min,y_min,w,h,类别号))相对/真实坐标
    tp = 0
    fn = 0
    for i in range(len(true)):
        target = true[i]
        iou_all = iou(pred, target)
        judge_tp = torch.where((iou_all > iou_threshold) & (pred[:, 4] == target[4]), True, False)
        judge_fn = torch.where((iou_all > iou_threshold) & (pred[:, 4] != target[4]), True, False)
        tp += len(pred[judge_tp])
        fn += len(pred[judge_fn])
    fp = len(pred) - tp - fn
    return tp, fn, fp
