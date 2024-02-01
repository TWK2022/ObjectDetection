import os
import cv2
import time
import torch
import argparse
import torchvision
import numpy as np
import albumentations
from model.layer import deploy

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='|pt模型推理|')
parser.add_argument('--model_path', default='best.pt', type=str, help='|pt模型位置|')
parser.add_argument('--image_path', default='image', type=str, help='|图片文件夹位置|')
parser.add_argument('--input_size', default=640, type=int, help='|模型输入图片大小|')
parser.add_argument('--batch', default=1, type=int, help='|输入图片批量|')
parser.add_argument('--confidence_threshold', default=0.35, type=float, help='|置信筛选度阈值(>阈值留下)|')
parser.add_argument('--iou_threshold', default=0.65, type=float, help='|iou阈值筛选阈值(<阈值留下)|')
parser.add_argument('--device', default='cuda', type=str, help='|推理设备|')
parser.add_argument('--num_worker', default=0, type=int, help='|CPU处理数据的进程数，0只有一个主进程，一般为0、2、4、8|')
parser.add_argument('--float16', default=False, type=bool, help='|推理数据类型，要支持float16的GPU，False时为float32|')
args, _ = parser.parse_known_args()  # 防止传入参数冲突，替代args = parser.parse_args()
args.model_path = args.model_path.split('.')[0] + '.pt'
# -------------------------------------------------------------------------------------------------------------------- #
# 初步检查
assert os.path.exists(args.model_path), f'! model_path不存在:{args.model_path} !'
assert os.path.exists(args.data_path), f'! data_path不存在:{args.data_path} !'
if args.float16:
    assert torch.cuda.is_available(), 'cuda不可用，因此无法使用float16'


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
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


def nms(pred, iou_threshold):  # 输入为(batch,(x_min,y_min,w,h))相对/真实坐标
    pred[:, 2:4] = pred[:, 0:2] + pred[:, 2:4]  # (x_min,y_min,x_max,y_max)真实坐标
    index = torchvision.ops.nms(pred[:, 0:4], pred[:, 4], 1 - iou_threshold)[:100]  # 非极大值抑制，最多100
    pred = pred[index]
    pred[:, 2:4] = pred[:, 2:4] - pred[:, 0:2]  # (x_min,y_min,w,h)真实坐标
    return pred


def draw(image, frame, cls, name):  # 输入(x_min,y_min,w,h)真实坐标
    image = image.astype(np.uint8)
    for i in range(len(frame)):
        a = (int(frame[i][0]), int(frame[i][1]))
        b = (int(frame[i][0] + frame[i][2]), int(frame[i][1] + frame[i][3]))
        cv2.rectangle(image, a, b, color=(0, 255, 0), thickness=2)
        cv2.putText(image, 'class:' + str(cls[i]), a, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite('save_' + name, image)
    print(f'| {name}: save_{name} |')


def test_pt(args):
    # 加载模型
    model_dict = torch.load(args.model_path, map_location='cpu')
    model = model_dict['model']
    model = deploy(model, args.input_size)
    model.half().eval().to(args.device) if args.float16 else model.float().eval().to(args.device)
    epoch = model_dict['epoch_finished']
    m_ap = round(model_dict['standard'], 4)
    print(f'| 模型加载成功:{args.model_path} | epoch:{epoch} | m_ap:{m_ap}|')
    # 推理
    image_dir = sorted(os.listdir(args.image_path))
    start_time = time.time()
    with torch.no_grad():
        dataloader = torch.utils.data.DataLoader(torch_dataset(image_dir), batch_size=args.batch, shuffle=False,
                                                 drop_last=False, pin_memory=False, num_workers=args.num_worker)
        for item, (image_batch, name_batch) in enumerate(dataloader):
            image_all = image_batch.cpu().numpy().astype(np.uint8)  # 转为numpy，用于画图
            image_batch = image_batch.to(args.device)
            pred_batch = model(image_batch)
            # 对batch中的每张图片分别操作
            for i in range(pred_batch[0].shape[0]):
                pred = [_[i] for _ in pred_batch]  # (Cx,Cy,w,h)
                pred = confidence_screen(pred, args.confidence_threshold)  # 置信度筛选
                if pred.shape[0] == 0:
                    print(f'{name_batch[i]}:None')
                    continue
                pred[:, 0:2] = pred[:, 0:2] - pred[:, 2:4] / 2  # (x_min,y_min,w,h)真实坐标
                pred = nms(pred, args.iou_threshold)  # 非极大值抑制
                frame = pred[:, 0:4]  # 边框
                cls = torch.argmax(pred[:, 5:], dim=1)  # 类别
                draw(image_all[i], frame.cpu().numpy(), cls.cpu().numpy(), name_batch[i])
    end_time = time.time()
    print('| 数据:{} 批量:{} 每张耗时:{:.4f} |'.format(len(image_dir), args.batch, (end_time - start_time) / len(image_dir)))


class torch_dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.transform = albumentations.Compose([
            albumentations.LongestMaxSize(args.input_size),
            albumentations.PadIfNeeded(min_height=args.input_size, min_width=args.input_size,
                                       border_mode=cv2.BORDER_CONSTANT, value=(128, 128, 128))])

    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, index):
        image = cv2.imread(args.image_path + '/' + self.image_dir[index])  # 读取图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为RGB通道
        image = self.transform(image=image)['image']  # 缩放和填充图片(归一化、调维度等在模型中完成)
        image = torch.tensor(image, dtype=torch.float16 if args.float16 else torch.float32)
        name = self.image_dir[index]
        return image, name


if __name__ == '__main__':
    test_pt(args)
