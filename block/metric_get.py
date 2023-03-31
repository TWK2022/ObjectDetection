import torch
import torchvision


def center_to_min(pred, true):  # (Cx,Cy)->(x_min,y_min)
    pred[:, 0:2] = pred[:, 0:2] - 1 / 2 * pred[:, 2:4]
    true[:, 0:2] = true[:, 0:2] - 1 / 2 * true[:, 2:4]
    return pred, true


def confidence_screen(pred, confidence_threshold):
    layer_num = len(pred)
    result = []
    for i in range(layer_num):  # 对一张图片的每个输出层分别进行操作
        judge = torch.where(pred[i][..., 4] > confidence_threshold, True, False)
        result.append((pred[i][judge]))
    result = torch.concat(result, dim=0)
    if result.shape[0] == 0:
        return result
    result = torch.stack(sorted(list(result), key=lambda x: x[4], reverse=True))  # 按置信度排序
    return result


def iou_single(A, B):  # 输入为(batch,(x_min,y_min,w,h))相对/真实坐标
    x1 = torch.maximum(A[:, 0], B[0])
    y1 = torch.maximum(A[:, 1], B[1])
    x2 = torch.minimum(A[:, 0] + A[:, 2], B[0] + B[2])
    y2 = torch.minimum(A[:, 1] + A[:, 3], B[1] + B[3])
    zeros = torch.zeros(1, device=A.device)
    intersection = torch.maximum(x2 - x1, zeros) * torch.maximum(y2 - y1, zeros)
    union = A[:, 2] * A[:, 3] + B[2] * B[3] - intersection
    return intersection / union


def iou(pred, true):  # 输入为(batch,(x_min,y_min,w,h))相对/真实坐标
    x1 = torch.maximum(pred[:, 0], true[:, 0])
    y1 = torch.maximum(pred[:, 1], true[:, 1])
    x2 = torch.minimum(pred[:, 0] + pred[:, 2], true[:, 0] + true[:, 2])
    y2 = torch.minimum(pred[:, 1] + pred[:, 3], true[:, 1] + true[:, 3])
    zeros = torch.zeros(1, device=pred.device)
    intersection = torch.maximum(x2 - x1, zeros) * torch.maximum(y2 - y1, zeros)
    union = pred[:, 2] * pred[:, 3] + true[:, 2] * true[:, 3] - intersection
    return intersection / union


def nms(pred, iou_threshold):  # 输入为(batch,(x_min,y_min,w,h))相对/真实坐标
    pred[:, 2:4] = pred[:, 0:2] + pred[:, 2:4]  # (x_min,y_min,x_max,y_max)真实坐标
    index = torchvision.ops.nms(pred[:, 0:4], pred[:, 4], iou_threshold)[:100]  # 非极大值抑制，最多100
    pred = pred[index]
    pred[:, 2:4] = pred[:, 2:4] - pred[:, 0:2]  # (x_min,y_min,w,h)真实坐标
    return pred
    # result = []
    # while len(pred) > 0:
    #     result.append(pred[0])  # 每轮开始时添加第一个到结果中
    #     pred = pred[1:]
    #     if len(pred) > 0:
    #         target = result[-1]
    #         iou_all = iou_single(pred, target)
    #         judge = torch.where(iou_all < iou_threshold, True, False)
    #         pred = pred[judge]
    # pred = torch.stack(result, dim=0)
    # return pred


def nms_tp_fn_fp(pred, true, iou_threshold):  # 输入为(batch,(x_min,y_min,w,h,其他,类别号))相对/真实坐标
    tp = 0
    fn = 0
    for i in range(len(true)):
        target = true[i]
        iou_all = iou_single(pred, target)
        judge_tp = torch.where((iou_all > iou_threshold) & (pred[:, 4] == target[4]), True, False)
        judge_fn = torch.where((iou_all > iou_threshold) & (pred[:, 4] != target[4]), True, False)
        tp += len(pred[judge_tp])  # 最多只有一个
        fn += len(pred[judge_fn])  # 最多只有一个
    fp = len(pred) - tp - fn
    return tp, fn, fp


def tp_tn_fp_fn(pred, true, judge, confidence_threshold, iou_threshold):  # 对网络输出(单个/批量)求非极大值抑制前的指标
    tp = 0
    tn = 0
    tp_fn = 0
    tn_fp = 0
    for i in range(len(pred)):
        if True in judge[i]:  # 有需要预测的位置
            pred_judge = pred[i][judge[i]]
            true_judge = true[i][judge[i]]
            pred_judge, true_judge = center_to_min(pred_judge, true_judge)  # Cx,Cy转为x_min,y_min
            judge_opposite = ~judge[i]  # True和False取反
            pred_confidence = pred_judge[..., 4]  # 需要预测的位置
            pred_confidence_opposite = pred[i][judge_opposite][..., 4]  # 不需要预测的位置
            pred_class = torch.argmax(pred_judge[..., 5:], dim=1)
            true_class = torch.argmax(pred_judge[..., 5:], dim=1)
            judge_tp = torch.where((pred_confidence >= confidence_threshold) & (pred_class == true_class) &
                                   (iou(pred_judge[..., 0:4], true_judge[..., 0:4]) > iou_threshold), True, False)
            judge_tn = torch.where(pred_confidence_opposite < confidence_threshold, True, False)
            tp_fn += len(pred_confidence)
            tp += len(pred_confidence[judge_tp])
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
