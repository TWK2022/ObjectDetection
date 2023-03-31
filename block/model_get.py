import os
import torch


def model_get(args):
    if os.path.exists(args.weight):
        model_dict = torch.load(args.weight, map_location='cpu')
    else:
        choice_dict = {'yolov5': 'model_prepare(args)._yolov5()',
                       'yolov7': 'model_prepare(args)._yolov7()'}
        model = eval(choice_dict[args.model])
        model_dict = {}
        model_dict['model'] = model
        model_dict['epoch'] = 0
        model_dict['optimizer_state_dict'] = None
        model_dict['ema_updates'] = 0
        model_dict['standard'] = 0
    return model_dict


class model_prepare(object):
    def __init__(self, args):
        self.args = args

    def _yolov5(self):
        from model.yolov5 import yolov5
        model = yolov5(self.args)
        return model

    def _yolov7(self):
        from model.yolov7 import yolov7
        model = yolov7(self.args)
        return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='分类任务')
    parser.add_argument('--model', default='', type=str, help='|模型选择|')
    parser.add_argument('--weight', default='', type=str, help='|模型位置，如果没找到模型则创建新模型|')
    args = parser.parse_args()
    model_get(args)
