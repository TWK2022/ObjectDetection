import torch


def loss_get(args):
    loss = loss_prepare(args)._load
    return loss


class loss_prepare(object):
    def __init__(self, args):
        self.loss_frame = self._ciou  # 边框损失函数
        self.loss_confidence = torch.nn.BCELoss()  # 置信度损失函数
        self.loss_confidence_target = torch.nn.BCELoss()
        self.loss_class = torch.nn.BCELoss()  # 分类损失函数
        self.loss_weight = args.loss_weight  # 每个输出层的权重

    def _load(self, pred, true, judge):  # pred与true的形式对应，judge为True和False组成的矩阵，True代表该位置有标签需要预测
        frame_loss = 0  # 总边框损失
        confidence_loss = 0  # 总置信度损失
        class_loss = 0  # 总分类损失
        for i in range(len(pred)):  # 对每个输出层分别进行操作
            if True in judge[i]:  # 有需要预测的位置
                pred_judge = pred[i][judge[i]]  # 预测的值
                true_judge = true[i][judge[i]]  # 真实的标签
                pred_judge, true_judge = self._center_to_min(pred_judge, true_judge)  # Cx,Cy转为x_min,y_min
                frame_add = self.loss_frame(pred_judge[:, 0:4], true_judge[:, 0:4])  # 边框损失(只计算需要的)
                confidence_add = self.loss_confidence(pred[i][..., 4], true[i][..., 4])  # 置信度损失(计算所有的)
                confidence_add += self.loss_confidence_target(pred_judge[i][..., 4], true_judge[i][..., 4]) * 0.2
                class_add = self.loss_class(pred_judge[:, 5:], true_judge[:, 5:])  # 分类损失(只计算需要的)
                frame_loss += self.loss_weight[i][0] * self.loss_weight[i][1] * (1 - torch.mean(frame_add))  # 总边框损失
                confidence_loss += self.loss_weight[i][0] * self.loss_weight[i][2] * confidence_add  # 总置信度损失
                class_loss += self.loss_weight[i][0] * self.loss_weight[i][3] * class_add  # 总分类损失
            else:  # 没有需要预测的位置
                confidence_add = self.loss_confidence(pred[i][..., 4], true[i][..., 4])  # 置信度损失(计算所有的)
                confidence_loss += self.loss_weight[i][0] * self.loss_weight[i][2] * confidence_add  # 总置信度损失
        return frame_loss + confidence_loss + class_loss, frame_loss, confidence_loss, class_loss

    def _center_to_min(self, pred, true):  # (Cx,Cy)->(x_min,y_min)
        pred[:, 0:2] = pred[:, 0:2] - 1 / 2 * pred[:, 2:4]
        true[:, 0:2] = true[:, 0:2] - 1 / 2 * true[:, 2:4]
        return pred, true

    def _ciou(self, pred, true):  # 输入为(batch,(x_min,y_min,w,h))相对/真实坐标
        iou = self._iou(pred, true)
        L1_L2 = self._L1_L2(pred, true)
        v = (4 / (3.14159 ** 2)) * torch.square(
            torch.atan(true[:, 2] / true[:, 3]) - torch.atan(pred[:, 2] / pred[:, 3]))
        with torch.no_grad():
            alpha = v / (1 - iou + v + 0.00001)
        return iou - L1_L2 - alpha * v

    def _iou(self, pred, true):  # 输入为(batch,(x_min,y_min,w,h))相对/真实坐标
        x1 = torch.maximum(pred[:, 0], true[:, 0])
        y1 = torch.maximum(pred[:, 1], true[:, 1])
        x2 = torch.minimum(pred[:, 0] + pred[:, 2], true[:, 0] + true[:, 2])
        y2 = torch.minimum(pred[:, 1] + pred[:, 3], true[:, 1] + true[:, 3])
        zeros = torch.zeros(1, device=pred.device)
        intersection = torch.maximum(x2 - x1, zeros) * torch.maximum(y2 - y1, zeros)
        union = pred[:, 2] * pred[:, 3] + true[:, 2] * true[:, 3] - intersection
        return intersection / union

    def _L1_L2(self, pred, true):  # 输入为(batch,(x_min,y_min,w,h))相对/真实坐标
        x1 = torch.minimum(pred[:, 0], true[:, 0])
        y1 = torch.minimum(pred[:, 1], true[:, 1])
        x2 = torch.maximum(pred[:, 0] + pred[:, 2], true[:, 0] + true[:, 2])
        y2 = torch.maximum(pred[:, 1] + pred[:, 3], true[:, 1] + true[:, 3])
        L1 = torch.square(pred[:, 0] - true[:, 0]) + torch.square(pred[:, 1] - true[:, 1])
        L2 = torch.square(x2 - x1) + torch.square(y2 - y1)
        return L1 / L2
