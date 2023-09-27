import torch


def loss_get(args):
    loss = loss_prepare(args)
    return loss


class loss_prepare(object):
    def __init__(self, args):
        self.loss_frame = self._ciou  # 边框损失函数
        self.loss_confidence = torch.nn.BCEWithLogitsLoss()  # 置信度损失函数
        self.loss_confidence_add = torch.nn.BCEWithLogitsLoss()  # 置信度正样本损失函数
        self.loss_class = torch.nn.BCEWithLogitsLoss()  # 分类损失函数
        self.loss_weight = args.loss_weight  # 每个输出层的权重
        self.stride = (8, 16, 32)
        # 预测值边框解码部分
        self.device = args.device
        output_size = [int(args.input_size // i) for i in self.stride]
        self.anchor = (((12, 16), (19, 36), (40, 28)), ((36, 75), (76, 55), (72, 146)),
                       ((142, 110), (192, 243), (459, 401)))
        self.grid = [0, 0, 0]
        for i in range(3):
            self.grid[i] = torch.arange(output_size[i]).to(args.device)

    def __call__(self, pred, true, judge):  # pred与true的形式对应，judge为True和False组成的矩阵，True代表该位置有标签需要预测
        frame_loss = 0  # 总边框损失
        confidence_loss = 0  # 总置信度损失
        class_loss = 0  # 总分类损失
        pred = self._frame_decode(pred)  # 将边框解码为(Cx,Cy,w,h)真实坐标。置信度和分类归一化在BCEWithLogitsLoss中完成
        for i in range(len(pred)):  # 对每个输出层分别进行操作
            if True in judge[i]:  # 有需要预测的位置
                pred_judge = pred[i][judge[i]]  # 预测的值
                true_judge = true[i][judge[i]]  # 真实的标签
                pred_judge, true_judge = self._center_to_min(pred_judge, true_judge)  # Cx,Cy转为x_min,y_min
                # 计算损失
                frame_add = self.loss_frame(pred_judge[:, 0:4], true_judge[:, 0:4])  # 边框损失(只计算需要的)
                confidence_a = 0.55 * self.loss_confidence(pred[i][..., 4], true[i][..., 4])  # 置信度损失(计算所有的)
                confidence_b = 0.45 * self.loss_confidence_add(pred_judge[i][..., 4], true_judge[i][..., 4])  # 正样本
                confidence_add = confidence_a + confidence_b
                class_add = self.loss_class(pred_judge[:, 5:], true_judge[:, 5:])  # 分类损失(只计算需要的)
                # 总损失
                frame_loss += self.loss_weight[i][0] * self.loss_weight[i][1] * (1 - torch.mean(frame_add))  # 总边框损失
                confidence_loss += self.loss_weight[i][0] * self.loss_weight[i][2] * confidence_add  # 总置信度损失
                class_loss += self.loss_weight[i][0] * self.loss_weight[i][3] * class_add  # 总分类损失
            else:  # 没有需要预测的位置
                confidence_add = self.loss_confidence(pred[i][..., 4], true[i][..., 4])  # 置信度损失(计算所有的)
                confidence_loss += self.loss_weight[i][0] * self.loss_weight[i][2] * confidence_add  # 总置信度损失
        return frame_loss + confidence_loss + class_loss, frame_loss, confidence_loss, class_loss

    def _frame_decode(self, output):
        # 遍历每一个大层
        for i in range(len(output)):
            output[i][..., 0:4] = output[i][..., 0:4].sigmoid()  # 归一化
            # 中心坐标[0-1]->[-0.5-1.5]->[-0.5*stride-80/40/20.5*stride]
            output[i][..., 0] = (2 * output[i][..., 0] - 0.5 + self.grid[i].unsqueeze(1)) * self.stride[i]
            output[i][..., 1] = (2 * output[i][..., 1] - 0.5 + self.grid[i]) * self.stride[i]
            # 遍历每一个大层中的小层
            for j in range(3):
                output[i][:, j, ..., 2] = 4 * output[i][:, j, ..., 2] ** 2 * self.anchor[i][j][0]  # [0-1]->[0-4*anchor]
                output[i][:, j, ..., 3] = 4 * output[i][:, j, ..., 3] ** 2 * self.anchor[i][j][1]  # [0-1]->[0-4*anchor]
        return output

    def _center_to_min(self, pred, true):  # (Cx,Cy)->(x_min,y_min)
        pred[:, 0:2] = pred[:, 0:2] - pred[:, 2:4] / 2
        true[:, 0:2] = true[:, 0:2] - true[:, 2:4] / 2
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
