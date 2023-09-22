# 根据yolov5改编:https://github.com/ultralytics/yolov5
import torch
from model.layer import cbs, c3, sppf, concat, head


class yolov5(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_size = args.input_size
        self.stride = (8, 16, 32)
        self.output_num = (3, 3, 3)  # 每个输出层的小层数
        self.output_size = [int(self.input_size // i) for i in self.stride]  # 每个输出层的尺寸，如(80,40,20)
        self.output_class = args.output_class
        dim_dict = {'n': 8, 's': 16, 'm': 32, 'l': 64}
        n_dict = {'n': 1, 's': 1, 'm': 2, 'l': 3}
        dim = dim_dict[args.model_type]
        n = n_dict[args.model_type]
        # 网络结构
        self.l0 = cbs(3, dim, 6, 2)  # 1/2
        self.l1 = cbs(dim, 2 * dim, 3, 2)  # 1/4
        # ---------- #
        self.l2 = c3(2 * dim, 2 * dim, n)
        self.l3 = cbs(2 * dim, 4 * dim, 3, 2)  # 1/8
        self.l4 = c3(4 * dim, 4 * dim, 2 * n)
        self.l5 = cbs(4 * dim, 8 * dim, 3, 2)  # 1/16
        self.l6 = c3(8 * dim, 8 * dim, 3 * n)
        self.l7 = cbs(8 * dim, 16 * dim, 3, 2)  # 1/32
        self.l8 = c3(16 * dim, 16 * dim, n)
        self.l9 = sppf(16 * dim, 16 * dim)
        self.l10 = cbs(16 * dim, 8 * dim, 1, 1)
        # ---------- #
        self.l11 = torch.nn.Upsample(scale_factor=2)  # 1/16
        self.l12 = concat(1)
        self.l13 = c3(16 * dim, 8 * dim, n)
        self.l14 = cbs(8 * dim, 4 * dim, 1, 1)
        # ---------- #
        self.l15 = torch.nn.Upsample(scale_factor=2)  # 1/8
        self.l16 = concat(1)
        self.l17 = c3(8 * dim, 4 * dim, n)  # 接output0
        # ---------- #
        self.l18 = cbs(4 * dim, 4 * dim, 3, 2)  # 1/16
        self.l19 = concat(1)
        self.l20 = c3(8 * dim, 8 * dim, n)  # 接output1
        # ---------- #
        self.l21 = cbs(8 * dim, 8 * dim, 3, 2)  # 1/32
        self.l22 = concat(1)
        self.l23 = c3(16 * dim, 16 * dim, n)  # 接output2
        # ---------- #
        self.output0 = head(4 * dim, self.output_size[0], self.output_class)
        self.output1 = head(8 * dim, self.output_size[1], self.output_class)
        self.output2 = head(16 * dim, self.output_size[2], self.output_class)

    def forward(self, x):
        x = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        l4 = self.l4(x)
        x = self.l5(l4)
        l6 = self.l6(x)
        x = self.l7(l6)
        x = self.l8(x)
        x = self.l9(x)
        l10 = self.l10(x)
        x = self.l11(l10)
        x = self.l12([x, l6])
        x = self.l13(x)
        l14 = self.l14(x)
        x = self.l15(l14)
        x = self.l16([x, l4])
        x = self.l17(x)
        output0 = self.output0(x)
        output0 = output0.reshape(-1, 3, self.output_size[0], self.output_size[0], 5 + self.output_class)  # 变形
        x = self.l18(x)
        x = self.l19([x, l14])
        x = self.l20(x)
        output1 = self.output1(x)
        output1 = output1.reshape(-1, 3, self.output_size[1], self.output_size[1], 5 + self.output_class)  # 变形
        x = self.l21(x)
        x = self.l22([x, l10])
        x = self.l23(x)
        output2 = self.output2(x)
        output2 = output2.reshape(-1, 3, self.output_size[2], self.output_size[2], 5 + self.output_class)  # 变形
        return [output0, output1, output2]


if __name__ == '__main__':
    import argparse
    from layer import cbs, c3, sppf, concat, head

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_type', default='n', type=str)
    parser.add_argument('--input_size', default=640, type=int)
    parser.add_argument('--output_class', default=1, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    args = parser.parse_args()
    model = yolov5(args).to(args.device)
    print(model)
    tensor = torch.rand(2, args.input_size, args.input_size, 3, dtype=torch.float32).to(args.device)
    pred = model(tensor)
    print(pred[0].shape, pred[1].shape, pred[2].shape)
