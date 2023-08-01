import torch
import torchvision


def center_to_min(pred, true):  # (Cx,Cy)->(x_min,y_min)
    pred[:, 0:2] = pred[:, 0:2] - 1 / 2 * pred[:, 2:4]
    true[:, 0:2] = true[:, 0:2] - 1 / 2 * true[:, 2:4]
    return pred, true


def confidence_screen(pred, confidence_threshold):
    result = []
    for i in range(len(pred)):  # 对一张图片的每个输出层分别进行操作
        judge = torch.where(pred[i][..., 4] > confidence_threshold, True, False)
        result.append((pred[i][judge]))
    result = torch.concat(result, dim=0)
    if result.shape[0] == 0:
        return result
    index = torch.argsort(result[:, 4], dim=0, descending=True)
    result = result[index]
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
    index = torchvision.ops.nms(pred[:, 0:4], pred[:, 4], 1 - iou_threshold)[:100]  # 非极大值抑制，最多100
    pred = pred[index]
    pred[:, 2:4] = pred[:, 2:4] - pred[:, 0:2]  # (x_min,y_min,w,h)真实坐标
    return pred


def nms_tp_fn_fp(pred, true, iou_threshold):  # 输入为(batch,(x_min,y_min,w,h,其他,类别号))相对/真实坐标
    pred_cls = torch.argmax(pred[:, 5:], dim=1)
    true_cls = torch.argmax(true[:, 5:], dim=1)
    tp = 0
    fn = 0
    for i in range(len(true)):
        target = true[i]
        iou_all = iou_single(pred, target)
        judge_tp = torch.where((iou_all > iou_threshold) & (pred_cls == true_cls[i]), True, False)
        judge_fn = torch.where((iou_all > iou_threshold) & (pred_cls != true_cls[i]), True, False)
        tp += len(pred[judge_tp])  # 最多只有一个
        fn += len(pred[judge_fn])  # 最多只有一个
    fp = len(pred) - tp - fn
    return tp, fp, fn
