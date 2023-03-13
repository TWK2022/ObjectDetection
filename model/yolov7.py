import torch
from model.layer import image_deal, cbs, elan, elan_h, mp, sppcspc, concat, head, decode


class yolov7(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.stride = (8, 16, 32)
        self.output_num = (3, 3, 3)  # 每个输出层的小层数
        self.anchor = (((10, 13), (16, 30), (33, 23)), ((30, 61), (62, 45), (59, 119)),
                       ((116, 90), (156, 198), (373, 326)))
        self.output_size = [int(args.input_size // i) for i in self.stride]  # 每个输出层的尺寸，如(80,40,20)
        self.output_class = args.output_class
        dim_dict = {'n': 8, 's': 16, 'm': 32, 'l': 64}
        n_dict = {'n': 1, 's': 1, 'm': 2, 'l': 3}
        dim = dim_dict[args.model_type]
        n = n_dict[args.model_type]
        # 解码网格
        self.grid = [0, 0, 0]
        for i in range(3):
            self.grid[i] = torch.arange(self.output_size[i])
        # 网络结构
        self.image_deal = image_deal()
        self.l0 = cbs(3, dim, 3, 1)
        self.l1 = cbs(dim, 2 * dim, 3, 2)  # 1/2
        self.l2 = cbs(2 * dim, 2 * dim, 3, 1)
        self.l3 = cbs(2 * dim, 4 * dim, 3, 2)  # 1/4
        # ---------- #
        self.l4 = elan(4 * dim, 8 * dim, n)
        self.l5 = mp(8 * dim, 8 * dim)  # 1/8
        self.l6 = elan(8 * dim, 16 * dim, n)
        self.l7 = mp(16 * dim, 16 * dim)  # 1/16
        self.l8 = elan(16 * dim, 32 * dim, n)
        self.l9 = mp(32 * dim, 32 * dim)  # 1/32
        self.l10 = elan(32 * dim, 32 * dim, n)
        self.l11 = sppcspc(32 * dim, 16 * dim)
        self.l12 = cbs(16 * dim, 8 * dim, 1, 1)
        # ---------- #
        self.l13 = torch.nn.Upsample(scale_factor=2)  # 1/16
        self.l8_add = cbs(32 * dim, 8 * dim, 1, 1)
        self.l14 = concat(1)
        self.l15 = elan_h(16 * dim, 8 * dim)
        self.l16 = cbs(8 * dim, 4 * dim, 1, 1)
        # ---------- #
        self.l17 = torch.nn.Upsample(scale_factor=2)  # 1/8
        self.l6_add = cbs(16 * dim, 4 * dim, 1, 1)
        self.l18 = concat(1)
        self.l19 = elan_h(8 * dim, 4 * dim)  # 接output0
        # ---------- #
        self.l20 = mp(4 * dim, 8 * dim)
        self.l21 = concat(1)
        self.l22 = elan_h(16 * dim, 8 * dim)  # 接output1
        # ---------- #
        self.l23 = mp(8 * dim, 16 * dim)
        self.l24 = concat(1)
        self.l25 = elan_h(32 * dim, 16 * dim)  # 接output2
        # ---------- #
        self.output0 = head(4 * dim, 3 * (5 + self.output_class))
        self.output1 = head(8 * dim, 3 * (5 + self.output_class))
        self.output2 = head(16 * dim, 3 * (5 + self.output_class))
        self.decode = decode(self.grid, self.stride, self.anchor)

    def forward(self, x):
        # 输入(batch,640,640,3)
        x = self.image_deal(x)
        x = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        l6 = self.l6(x)
        x = self.l7(l6)
        l8 = self.l8(x)
        x = self.l9(l8)
        x = self.l10(x)
        l11 = self.l11(x)
        x = self.l12(l11)
        x = self.l13(x)
        l8_add = self.l8_add(l8)
        x = self.l14([x, l8_add])
        l15 = self.l15(x)
        x = self.l16(l15)
        x = self.l17(x)
        l6_add = self.l6_add(l6)
        x = self.l18([x, l6_add])
        x = self.l19(x)
        output0 = self.output0(x)
        output0 = output0.reshape(-1, 3, self.output_size[0], self.output_size[0], 5 + self.output_class)  # 变形
        x = self.l20(x)
        x = self.l21([x, l15])
        x = self.l22(x)
        output1 = self.output1(x)
        output1 = output1.reshape(-1, 3, self.output_size[1], self.output_size[1], 5 + self.output_class)  # 变形
        x = self.l23(x)
        x = self.l24([x, l11])
        x = self.l25(x)
        output2 = self.output2(x)
        output2 = output2.reshape(-1, 3, self.output_size[2], self.output_size[2], 5 + self.output_class)  # 变形
        output = self.decode([output0, output1, output2])
        return output


if __name__ == '__main__':
    import argparse
    from layer import image_deal, cbs, elan, elan_h, mp, sppcspc, concat, head, decode

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_type', default='n', type=str)
    parser.add_argument('--input_size', default=640, type=int)
    parser.add_argument('--output_class', default=1, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    args = parser.parse_args()
    model = yolov7(args).to(args.device)
    print(model)
    tensor = torch.rand(2, args.input_size, args.input_size, 3, dtype=torch.float32).to(args.device)
    pred = model(tensor)
    print(pred[0].shape, pred[1].shape, pred[2].shape)
