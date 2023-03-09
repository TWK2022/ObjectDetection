import os
import cv2
import time
import torch
import argparse
import numpy as np
import albumentations

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='pt模型推理')
parser.add_argument('--model_path', default='best.pt', type=str, help='|pt模型位置|')
parser.add_argument('--image_path', default='image', type=str, help='|图片文件夹位置|')
parser.add_argument('--input_size', default=640, type=int, help='|模型输入图片大小|')
parser.add_argument('--batch', default=1, type=int, help='|输入图片批量|')
parser.add_argument('--confidence_threshold', default=0.8, type=float, help='|置信筛选度阈值(>阈值留下)|')
parser.add_argument('--iou_threshold', default=0.65, type=float, help='|iou阈值筛选阈值(>阈值留下)|')
parser.add_argument('--device', default='cuda', type=str, help='|用CPU/GPU推理|')
parser.add_argument('--float16', default=True, type=bool, help='|推理数据类型，要支持float16的GPU，False时为float32|')
args = parser.parse_args()
args.model_path = args.model_path.split('.')[0] + '.pt'
# -------------------------------------------------------------------------------------------------------------------- #
# 初步检查
assert os.path.exists(args.model_path), f'没有找到模型{args.model_path}'
assert os.path.exists(args.image_path), f'没有找到图片文件夹{args.image_path}'
if args.float16:
    assert torch.cuda.is_available(), 'cuda不可用，因此无法使用float16'


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
def confidence_screen(pred, confidence_threshold):
    layer_num = len(pred)
    result = []
    for i in range(layer_num):  # 对一张图片的每个输出层分别进行操作
        judge = np.where(pred[i][..., 4] > confidence_threshold, True, False)
        result.append((pred[i][judge]))
    result = np.concatenate(result, axis=0)
    return result


def iou(A, B):  # 输入为(batch,(x_min,y_min,w,h))
    x1 = np.maximum(A[:, 0], B[0])
    y1 = np.maximum(A[:, 1], B[1])
    x2 = np.minimum(A[:, 0] + A[:, 2], B[0] + B[2])
    y2 = np.minimum(A[:, 1] + A[:, 3], B[1] + B[3])
    zeros = np.zeros(1)
    intersection = np.maximum(x2 - x1, zeros) * np.maximum(y2 - y1, zeros)
    union = A[:, 2] * A[:, 3] + B[2] * B[3] - intersection
    return intersection / union


def nms(pred, iou_threshold):
    choose = np.stack(sorted(list(pred), key=lambda x: x[4], reverse=True))  # 待选择的预测值
    result = []
    while len(choose) > 0:
        result.append(choose[0])  # 每轮开始时添加第一个到结果中
        choose = choose[1:]
        if len(choose) > 0:
            target = result[-1]
            iou_all = iou(choose, target)
            judge = np.where(iou_all < iou_threshold, True, False)
            choose = choose[judge]
    return np.stack(result, axis=0)


def draw(image, frame, cls, name):
    for i in range(len(frame)):
        a = (int(frame[i][0] - frame[i][2] / 2), int(frame[i][1] - frame[i][3] / 2))
        b = (int(frame[i][0] + frame[i][2] / 2), int(frame[i][1] + frame[i][3] / 2))
        cv2.rectangle(image, a, b, color=(0, 255, 0), thickness=2)
        cv2.putText(image, 'class:' + str(cls[i]), a, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite('pred_' + name, image)
    cv2.imshow('pred_' + name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_pt():
    # 加载模型
    model_dict = torch.load(args.model_path, map_location='cpu')
    model = model_dict['model']
    model.half().eval().to(args.device) if args.float16 else model.float().eval().to(args.device)
    print('| 模型加载成功:{} |'.format(args.model_path))
    # 推理
    image_dir = sorted(os.listdir(args.image_path))
    start_time = time.time()
    with torch.no_grad():
        dataloader = torch.utils.data.DataLoader(torch_dataset(image_dir), batch_size=args.batch,
                                                 shuffle=False, drop_last=False, pin_memory=False)
        for item, (image_batch, name_batch) in enumerate(dataloader):
            image_all = image_batch.cpu().numpy().astype(np.uint8)  # 转为numpy，用于画图
            image_batch = image_batch.to(args.device)
            pred_batch = model(image_batch)
            pred_batch = [pred_batch[i].cpu().numpy() for i in range(len(pred_batch))]  # 转为numpy
            # 对batch中的每张图片分别操作
            for i in range(args.batch):
                pred = [pred_batch[j][i] for j in range(len(pred_batch))]
                pred = confidence_screen(pred, args.confidence_threshold)  # 置信度筛选
                pred[0:2] = pred[0:2] - 1 / 2 * pred[2:4]
                pred = nms(pred, args.iou_threshold)  # 非极大值抑制
                frame = pred[:, 0:4]  # 边框
                cls = np.argmax(pred[:, 5:], axis=1)  # 类别
                draw(image_all[i], frame, cls, name_batch[i])
    end_time = time.time()
    print('| 数据:{} 批量:{} 每张耗时:{:.4f} |'.format(len(image_dir), args.batch, (end_time - start_time) / len(image_dir)))


class torch_dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.transform = albumentations.Compose([
            albumentations.LongestMaxSize(args.input_size),
            albumentations.PadIfNeeded(min_height=args.input_size, min_width=args.input_size,
                                       border_mode=cv2.BORDER_CONSTANT, value=(127, 127, 127))])

    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, index):
        image = cv2.imread(args.image_path + '/' + self.image_dir[index])  # 读取图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为RGB通道
        image = self.transform(image=image)['image']  # 缩放和填充图片(归一化、减均值、除以方差、调维度等在模型中完成)
        image = torch.tensor(image, dtype=torch.float16 if args.float16 else torch.float32)
        name = self.image_dir[index]
        return image, name


if __name__ == '__main__':
    test_pt()
