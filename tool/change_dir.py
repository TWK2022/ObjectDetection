import argparse

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='更改yolo格式数据集train.txt和val.txt中图片的路径')
parser.add_argument('--data_path', default=r'D:\dataset\ObjectDetection\voc', type=str, help='|数据根目录所在目录|')
parser.add_argument('--change_dir', default=r'D:\dataset\ObjectDetection\voc', type=str, help='|将路径中目录换成change_dir|')
args = parser.parse_args()
args.train_txt = args.data_path + '/train.txt'
args.val_txt = args.data_path + '/val.txt'
args.txt_change = args.change_dir + '/image'


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
def change_dir(txt):
    with open(txt, 'r')as f:
        label = f.readlines()
        label = [args.txt_change + _.split('image')[-1] for _ in label]
    with open(txt, 'w')as f:
        f.writelines(label)


if __name__ == '__main__':
    change_dir(args.train_txt)
    change_dir(args.val_txt)
    print(f'| 已更改train.txt和val.txt中的图片根路径为:{args.change_dir} |')
