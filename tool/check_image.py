import os
import argparse

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='检查train.txt和val.txt中图片是否存在')
parser.add_argument('--data_path', default=r'D:\dataset\ObjectDetection\voc', type=str, help='|图片所在目录|')
args = parser.parse_args()
args.train_path = args.data_path + '/train.txt'
args.val_path = args.data_path + '/val.txt'


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
def check_image(txt_path):
    with open(txt_path, 'r')as f:
        image_path = [_.strip() for _ in f.readlines()]
    for i in range(len(image_path)):
        if not os.path.exists(image_path[i]):
            print(f'| {txt_path}:不存在{image_path[i]} |')


if __name__ == '__main__':
    check_image(args.train_path)
    check_image(args.val_path)
    print(f'| 已完成{args.data_path}中train.txt和val.txt所需要的图片检擦 |')
