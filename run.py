# 数据需准备成以下格式(标准YOLO格式)
# ├── 数据集路径:data_path
#     └── image:存放所有图片
#     └── label:存放所有图片的标签，名称:图片名.txt，内容:类别号 x_center y_center w h(相对图片的比例值)
#     └── train.txt:训练图片的绝对路径(或相对data_path下路径)
#     └── val.txt:验证图片的绝对路径(或相对data_path下路径)
#     └── class.txt:所有的类别名称
# class.csv内容如下:
# 类别1
# 类别2
# ...
# -------------------------------------------------------------------------------------------------------------------- #
import os
import wandb
import torch
import argparse
from block.data_get import data_get
from block.model_get import model_get
from block.loss_get import loss_get
from block.train_get import train_get

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='目标检测')
parser.add_argument('--data_path', default=r'D:\dataset\ObjectDetection\lamp', type=str, help='|数据根目录路径|')
parser.add_argument('--wandb', default=False, type=bool, help='|是否使用wandb可视化|')
parser.add_argument('--wandb_project', default='test', type=str, help='|wandb项目名称|')
parser.add_argument('--wandb_name', default='train', type=str, help='|wandb项目中的训练名称|')
parser.add_argument('--wandb_image_num', default=16, type=int, help='|wandb保存图片的数量|')
parser.add_argument('--save_name', default='best.pt', type=str, help='|保存模型的位置|')
parser.add_argument('--weight', default='best.pt', type=str, help='|已有模型的位置，如果没找到模型则会创建新模型|')
parser.add_argument('--model', default='yolov5', type=str, help='|模型选择|')
parser.add_argument('--model_type', default='s', type=str, help='|模型的型号参数，部分模型有|')
parser.add_argument('--input_size', default=640, type=int, help='|输入图片大小|')
parser.add_argument('--output_class', default=1, type=int, help='|输出的类别数|')
parser.add_argument('--loss_weight', default=((1 / 3, 0.2, 0.6, 0.2), (1 / 3, 0.3, 0.5, 0.2), (1 / 3, 0.4, 0.4, 0.2)),
                    type=tuple, help='|每个输出层(从大到小排序)的权重->[总权重、边框权重、置信度权重、分类权重]|')
parser.add_argument('--label_smooth', default=(0.01, 0.99), type=tuple, help='|标签平滑的值|')
parser.add_argument('--epoch', default=50, type=int, help='|训练轮数|')
parser.add_argument('--batch', default=4, type=int, help='|训练批量大小|')
parser.add_argument('--lr', default=0.001, type=int, help='|初始学习率，训练中采用adam算法|')
parser.add_argument('--device', default='cuda', type=str, help='|训练设备|')
parser.add_argument('--latch', default=True, type=bool, help='|模型和数据是否为锁存，True为锁存|')
parser.add_argument('--num_worker', default=0, type=int, help='|CPU在处理数据时使用的进程数，0表示只有一个主进程，一般为0、2、4、8|')
parser.add_argument('--scaler', default=False, type=bool, help='|混合float16精度训练|')
parser.add_argument('--mosaic', default=0.5, type=float, help='|使用mosaic增强的概率|')
parser.add_argument('--confidence_threshold', default=0.5, type=float, help='|指标计算置信度阈值|')
parser.add_argument('--iou_threshold', default=0.5, type=float, help='|指标计算iou阈值|')
args = parser.parse_args()
# 为CPU设置随机种子
torch.manual_seed(999)
# 为所有GPU设置随机种子
torch.cuda.manual_seed_all(999)
# 固定每次返回的卷积算法
torch.backends.cudnn.deterministic = True
# cuDNN使用非确定性算法
torch.backends.cudnn.enabled = True
# 训练前cuDNN会先搜寻每个卷积层最适合实现它的卷积算法，加速运行；但对于复杂变化的输入数据，可能会有过长的搜寻时间，对于训练比较快的网络建议设为False
torch.backends.cudnn.benchmark = False
# wandb可视化:https://wandb.ai
if args.wandb:
    args.wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_name, config=args)
# 混合float16精度训练
if args.scaler:
    args.scaler = torch.cuda.amp.GradScaler()
# -------------------------------------------------------------------------------------------------------------------- #
# 初步检查
assert os.path.exists(args.data_path + '/' + 'image'), 'data_path中缺少image'
assert os.path.exists(args.data_path + '/' + 'label'), 'data_path中缺少label'
assert os.path.exists(args.data_path + '/' + 'train.txt'), 'data_path中缺少train.txt'
assert os.path.exists(args.data_path + '/' + 'val.txt'), 'data_path中缺少val.txt'
assert os.path.exists(args.data_path + '/' + 'class.txt'), 'data_path中缺少class.txt'
if os.path.exists(args.weight):  # 优先加载已有模型args.weight继续训练
    print('| 加载已有模型:{} |'.format(args.weight))
else:  # 创建自定义模型args.model
    assert os.path.exists('model/' + args.model + '.py'), '没有此自定义模型'.format(args.model)
    print('| 创建自定义模型:{} | 型号:{} |'.format(args.model, args.model_type))
if args.device.lower() in ['cuda', 'gpu']:  # 检查训练设备
    assert torch.cuda.is_available(), 'GPU不可用'
    args.device = 'cuda'
else:
    args.device = 'cpu'
print('| args:{} |'.format(args))
# -------------------------------------------------------------------------------------------------------------------- #
# 程序
if __name__ == '__main__':
    # 数据(只是图片路径和原始txt中标签，读取和预处理在训练/验证中完成)
    data_dict = data_get(args)
    # 模型
    model_dict = model_get(args)
    # 损失
    loss = loss_get(args)
    # 摘要
    print('| 训练集:{} | 验证集:{} | 模型:{} | 输入尺寸:{} | 初始学习率:{} |'
          .format(len(data_dict['train']), len(data_dict['val']), args.model, args.input_size, args.lr))
    # 训练(包括图片读取和预处理、训练、验证、保存模型)
    model_dict = train_get(args, data_dict, model_dict, loss)
    # 显示结果
    try:
        print('\n| 最佳结果 | train_loss:{:.4f} | train_frame_loss:{:.4f} | train_confidence_loss:{:.4f} |'
              ' train_class_loss:{:.4f} | val_loss:{:.4f} | val_frame_loss:{:.4f} | val_confidence_loss:{:.4f} |'
              ' val_class_loss:{:.4f} | val_accuracy:{:.4f} | val_precision:{:.4f} | val_recall:{:.4f} |'
              ' val_m_ap:{:.4f} | val_nms_precision:{:.4f} | val_nms_recall:{:.4f} | val_nms_m_ap:{:.4f} |\n'
              .format(model_dict['train_loss'], model_dict['train_frame_loss'], model_dict['train_confidence_loss'],
                      model_dict['train_class_loss'], model_dict['val_loss'], model_dict['val_frame_loss'],
                      model_dict['val_confidence_loss'], model_dict['val_class_loss'], model_dict['val_accuracy'],
                      model_dict['val_precision'], model_dict['val_recall'], model_dict['val_m_ap'],
                      model_dict['val_nms_precision'], model_dict['val_nms_recall'], model_dict['val_nms_m_ap']))
    except:
        print('\n| !由于指标太低没有保存最佳模型! |\n')
