import os
import cv2
import argparse
import onnxruntime
import numpy as np
import albumentations

# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|模型预测|')
parser.add_argument('--model_path', default='best.onnx', type=str, help='|模型位置|')
parser.add_argument('--image_dir', default='image', type=str, help='|图片文件夹位置|')
parser.add_argument('--input_size', default=640, type=int, help='|模型输入图片大小|')
parser.add_argument('--device', default='cpu', type=str, help='|设备|')
parser.add_argument('--float16', default=True, type=bool, help='|数据类型|')
parser.add_argument('--confidence_threshold', default=0.5, type=float, help='|置信度阈值|')
parser.add_argument('--iou_threshold', default=0.5, type=float, help='|iou阈值|')
args, _ = parser.parse_known_args()  # 防止传入参数冲突，替代args = parser.parse_args()


# -------------------------------------------------------------------------------------------------------------------- #
class predict_class:
    def __init__(self, args=args):
        self.args = args
        self.device = args.device
        self.float16 = args.float16
        self.confidence_threshold = args.confidence_threshold
        self.iou_threshold = args.iou_threshold
        provider = 'CUDAExecutionProvider' if args.device.lower() in ['gpu', 'cuda'] else 'CPUExecutionProvider'
        self.model = onnxruntime.InferenceSession(args.model_path, providers=[provider])  # 加载模型和框架
        self.input_name = self.model.get_inputs()[0].name  # 获取输入名称
        self.output_name = self.model.get_outputs()[0].name  # 获取输出名称
        self.transform = albumentations.Compose([
            albumentations.LongestMaxSize(args.input_size),
            albumentations.PadIfNeeded(min_height=args.input_size, min_width=args.input_size,
                                       border_mode=cv2.BORDER_CONSTANT, value=(128, 128, 128))])

    @staticmethod
    def iou(pred, label):  # 输入(batch,(x_min,y_min,w,h))相对/真实坐标
        x1 = np.maximum(pred[:, 0], label[:, 0])
        y1 = np.maximum(pred[:, 1], label[:, 1])
        x2 = np.minimum(pred[:, 0] + pred[:, 2], label[:, 0] + label[:, 2])
        y2 = np.minimum(pred[:, 1] + pred[:, 3], label[:, 1] + label[:, 3])
        intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        union = pred[:, 2] * pred[:, 3] + label[:, 2] * label[:, 3] - intersection
        return intersection / union

    def predict(self, image_dir=args.image_dir):
        image_name_list = sorted(os.listdir(image_dir))
        image_path_list = [f'{image_dir}/{_}' for _ in image_name_list]
        result = []
        for path in image_path_list:
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)  # 读取图片
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为RGB通道
            array = self.image_process(image)
            output = self.model.run([self.output_name], {self.input_name: array})[0][0]
            output = self.decode(output)
            result.append(len(output))
            image = self.transform(image=image)['image']  # 缩放和填充图片
            self.draw_image(image, output[:, 0:4], save_path=f'predict_{os.path.basename(path)}')
        print(result)

    def image_process(self, image):
        image = self.transform(image=image)['image']  # 缩放和填充图片
        image = image.astype(dtype=np.float16 if self.float16 else np.float32)
        image = image / 255
        image = image[np.newaxis].transpose(0, 3, 1, 2)
        return image

    def decode(self, pred):  # (cx,cy,w,h)
        pred[:, 0:2] = pred[:, 0:2] - pred[:, 2:4] / 2  # (x_min,y_min,w,h)
        pred = pred[pred[:, 4] > self.confidence_threshold]  # 置信度筛选
        pred = pred[np.where((pred[:, 0] > 0) & (pred[:, 1] > 0))]  # 去除负值
        if len(pred) == 0:  # 没有预测值
            return None
        pred = pred[np.argsort(pred[:, 4])[::-1]]  # 按置信度从大到小排序
        result = []
        while len(pred):
            result.append(pred[0])
            pred = pred[1:]
            iou = self.iou(result[-1][np.newaxis], pred)
            pred = pred[iou < self.iou_threshold]
        result = np.stack(result)
        result[:, 0:2] = result[:, 0:2] + result[:, 2:4] / 2  # (cx,cy,w,h)
        return result

    def draw_image(self, image, frame_all, save_path='draw.jpg'):  # 画图(cx,cy,w,h)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        frame_all[:, 0:2] = frame_all[:, 0:2] - frame_all[:, 2:4] / 2
        frame_all[:, 2:4] = frame_all[:, 0:2] + frame_all[:, 2:4]  # (x_min,y_min,x_max,y_max)
        for frame in frame_all:
            x1, y1, x2, y2 = frame
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255, 0), thickness=2)
        cv2.imwrite(save_path, image)


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    model = predict_class()
    model.predict()
