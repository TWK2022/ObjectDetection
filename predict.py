import os
import cv2
import torch
import argparse
import numpy as np
from model.layer import deploy

# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|模型预测|')
parser.add_argument('--model_path', default='best.pt', type=str, help='|模型位置|')
parser.add_argument('--image_path', default='image/test.jpg', type=str, help='|图片位置|')
parser.add_argument('--input_size', default=640, type=int, help='|模型输入图片大小|')
parser.add_argument('--batch', default=1, type=int, help='|输入图片批量|')
parser.add_argument('--device', default='cuda', type=str, help='|设备|')
parser.add_argument('--float16', default=True, type=bool, help='|数据类型|')
parser.add_argument('--confidence_threshold', default=0.5, type=float, help='|置信度阈值|')
parser.add_argument('--iou_threshold', default=0.5, type=float, help='|iou阈值|')
args, _ = parser.parse_known_args()  # 防止传入参数冲突，替代args = parser.parse_args()
args.device = 'cpu' if not torch.cuda.is_available() else args.device
args.float16 = True if torch.cuda.is_available() else False  # cpu使用float16会自动转为float32，但速度会很慢


# -------------------------------------------------------------------------------------------------------------------- #
class predict_class:
    def __init__(self, args=args):
        self.device = args.device
        self.float16 = args.float16
        self.input_size = args.input_size
        self.confidence_threshold = args.confidence_threshold
        self.iou_threshold = args.iou_threshold
        model_dict = torch.load(args.model_path, map_location='cpu', weights_only=False)
        model = deploy(model_dict['model'])
        self.model = model.half().eval().to(args.device) if args.float16 else model.float().eval().to(args.device)

    @staticmethod
    def iou(pred, label):  # 输入(batch,(x_min,y_min,w,h))相对/真实坐标
        x1 = np.maximum(pred[:, 0], label[:, 0])
        y1 = np.maximum(pred[:, 1], label[:, 1])
        x2 = np.minimum(pred[:, 0] + pred[:, 2], label[:, 0] + label[:, 2])
        y2 = np.minimum(pred[:, 1] + pred[:, 3], label[:, 1] + label[:, 3])
        intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        union = pred[:, 2] * pred[:, 3] + label[:, 2] * label[:, 3] - intersection
        return intersection / union

    @staticmethod
    def draw_image(image, frame_all, save_path='draw.jpg'):  # 画图(cx,cy,w,h)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        frame_all[:, 0:2] = frame_all[:, 0:2] - frame_all[:, 2:4] / 2
        frame_all[:, 2:4] = frame_all[:, 0:2] + frame_all[:, 2:4]  # (x_min,y_min,x_max,y_max)
        for frame in frame_all:
            x1, y1, x2, y2 = frame[0:4]
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255, 0), thickness=2)
        cv2.imwrite(save_path, image)

    def predict(self, image_path):
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)  # 读取图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为RGB通道
        tensor = torch.tensor(self.image_process(image)).to(self.device)
        with torch.no_grad():
            output = self.model(tensor).detach().cpu().numpy()[0]
        output = self.decode(output, image.shape)
        return output

    def image_process(self, image):
        image = cv2.resize(image, (self.input_size, self.input_size))
        image = image.astype(dtype=np.float16 if self.float16 else np.float32)
        image = image / 255
        image = image[np.newaxis].transpose(0, 3, 1, 2)
        return image

    def decode(self, pred, shape):  # (cx,cy,w,h)
        pred[:, 0:2] = pred[:, 0:2] - pred[:, 2:4] / 2  # (x_min,y_min,w,h)
        pred = pred[pred[:, 4] > self.confidence_threshold]  # 置信度筛选
        pred = pred[np.where((pred[:, 0] > 0) & (pred[:, 1] > 0))]  # 去除负值
        if len(pred) == 0:  # 没有预测值
            return None
        pred = self.nms(pred)
        pred[:, 0:2] = pred[:, 0:2] + pred[:, 2:4] / 2  # (cx,cy,w,h)
        x_cale, y_scale = shape[1] / self.input_size, shape[0] / self.input_size
        pred[:, 0] *= x_cale
        pred[:, 1] *= y_scale
        pred[:, 2] *= x_cale
        pred[:, 3] *= y_scale
        return pred

    def nms(self, pred):  # (x_min,y_min,w,h)
        score = pred[:, 4] * np.max(pred[:, 5:], axis=1)  # 综合置信度和类别筛选
        pred = pred[np.argsort(score)[::-1]].astype(np.float32)  # 按置信度从大到小排序，提高精度防止数据溢出
        result = []
        while len(pred):
            result.append(pred[0])
            pred = pred[1:]
            iou = self.iou(result[-1][np.newaxis], pred)
            pred = pred[iou < self.iou_threshold]
        result = np.stack(result)
        return result


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    model = predict_class()
    image = cv2.imdecode(np.fromfile(args.image_path, dtype=np.uint8), cv2.IMREAD_COLOR)  # 读取图片
    result = model.predict(image)
    if result is not None:
        model.draw_image(image, result[:, 0:4], save_path=f'predict_{os.path.basename(args.image_path)}')
    else:
        cv2.imwrite(f'predict_{os.path.basename(args.image_path)}', image)
