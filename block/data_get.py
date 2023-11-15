import numpy as np


def data_get(args):
    data_dict = data_prepare(args).load()
    return data_dict


class data_prepare(object):
    def __init__(self, args):
        self.args = args

    def load(self):
        data_dict = {}
        data_dict['train'] = self._load_label('train.txt')
        data_dict['val'] = self._load_label('val.txt')
        data_dict['class'] = self._load_class()
        return data_dict

    def _load_label(self, txt_name):
        with open(f'{self.args.data_path}/{txt_name}', encoding='utf-8')as f:
            txt = [_.strip() for _ in f.readlines()]  # 读取所有图片路径
        data_list = [[0, 0] for _ in range(len(txt))]  # [图片路径,原始标签]
        for i in range(len(txt)):
            image_path = f'{self.args.data_path}/image' + txt[i].split('image')[-1]
            data_list[i][0] = image_path
            with open(f'{self.args.data_path}/label/' + image_path.split('/')[-1].split('.')[0] + '.txt', 'r') as f:
                label_txt = [_.strip().split(' ') for _ in f.readlines()]  # 读取该图片的标签
            data_list[i][1] = np.array(label_txt, dtype=np.float32)
        return data_list

    def _load_class(self):
        with open(f'{self.args.data_path}/class.txt', encoding='utf-8')as f:
            txt = [_.strip() for _ in f.readlines()]
        return txt


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', default=r'D:\dataset\ObjectDetection\voc', type=str)
    parser.add_argument('--input_size', default=640, type=int)
    args = parser.parse_args()
    data_dict = data_get(args)
