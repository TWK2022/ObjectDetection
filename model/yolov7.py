# 根据yolov7改编:https://github.com/WongKinYiu/yolov7
import torch
from model.layer import cbs, elan, elan_h, mp, sppcspc, concat, head, decode


class yolov7(torch.nn.Module):
    def __init__(self, args, config=None):
        super().__init__()
        dim_dict = {'s': 16, 'm': 32, 'l': 64}
        n_dict = {'s': 1, 'm': 2, 'l': 3}
        dim = dim_dict[args.model_type]
        n = n_dict[args.model_type]
        self.output_size = args.output_size
        self.output_class = args.output_class
        self.decode = decode(args)  # 输出解码
        # 网络结构
        if config is None:  # 正常版本
            self.cbs0 = cbs(3, dim, 3, 1)
            self.cbs1 = cbs(dim, 2 * dim, 3, 2)  # input_size/2
            self.cbs2 = cbs(2 * dim, 2 * dim, 3, 1)
            self.cbs3 = cbs(2 * dim, 4 * dim, 3, 2)  # input_size/4
            self.elan4 = elan(4 * dim, 8 * dim, n)
            self.mp5 = mp(8 * dim, 8 * dim)  # input_size/8
            self.elan6 = elan(8 * dim, 16 * dim, n)
            self.mp7 = mp(16 * dim, 16 * dim)  # input_size/16
            self.elan8 = elan(16 * dim, 32 * dim, n)
            self.mp9 = mp(32 * dim, 32 * dim)  # input_size/32
            self.elan10 = elan(32 * dim, 32 * dim, n)
            self.sppcspc11 = sppcspc(32 * dim, 16 * dim)
            self.cbs12 = cbs(16 * dim, 8 * dim, 1, 1)
            # -------------------- #
            self.upsample13 = torch.nn.Upsample(scale_factor=2)  # input_size/16
            self.l8_add = cbs(32 * dim, 8 * dim, 1, 1)
            self.concat14 = concat(1)
            self.elan_h15 = elan_h(16 * dim, 8 * dim)
            self.cbs16 = cbs(8 * dim, 4 * dim, 1, 1)
            # -------------------- #
            self.upsample17 = torch.nn.Upsample(scale_factor=2)  # input_size/8
            self.l6_add = cbs(16 * dim, 4 * dim, 1, 1)
            self.concat18 = concat(1)
            self.elan_h19 = elan_h(8 * dim, 4 * dim)  # 接output0
            # -------------------- #
            self.mp20 = mp(4 * dim, 8 * dim)
            self.concat21 = concat(1)
            self.elan_h22 = elan_h(16 * dim, 8 * dim)  # 接output1
            # -------------------- #
            self.mp23 = mp(8 * dim, 16 * dim)
            self.concat24 = concat(1)
            self.elan_h25 = elan_h(32 * dim, 16 * dim)  # 接output2
            # -------------------- #
            self.head0 = head(4 * dim, self.output_size[0], self.output_class)
            self.head1 = head(8 * dim, self.output_size[1], self.output_class)
            self.head2 = head(16 * dim, self.output_size[2], self.output_class)
        else:  # 剪枝版本
            self.cbs0 = cbs(3, config[0], 1, 1)
            self.cbs1 = cbs(config[0], config[1], 3, 2)  # input_size/2
            self.cbs2 = cbs(config[1], config[2], 1, 1)
            self.cbs3 = cbs(config[2], config[3], 3, 2)  # input_size/4
            self.elan4 = elan(config[3], None, n, config[4:7 + 2 * n])
            self.mp5 = mp(config[6 + 2 * n], None, config[7 + 2 * n:10 + 2 * n])  # input_size/8
            self.elan6 = elan(config[7 + 2 * n] + config[9 + 2 * n], None, n, config[10 + 2 * n:13 + 4 * n])
            self.mp7 = mp(config[12 + 4 * n], None, config[13 + 4 * n:16 + 4 * n])  # input_size/16
            self.elan8 = elan(config[13 + 4 * n] + config[15 + 4 * n], None, n, config[16 + 4 * n:19 + 6 * n])
            self.mp9 = mp(config[18 + 6 * n], None, config[19 + 6 * n:22 + 6 * n])  # input_size/32
            self.elan10 = elan(config[19 + 6 * n] + config[21 + 6 * n], None, n, config[22 + 6 * n:25 + 8 * n])
            self.sppcspc11 = sppcspc(config[24 + 8 * n], None, config[25 + 8 * n:32 + 8 * n])
            self.cbs12 = cbs(config[31 + 8 * n], config[32 + 8 * n], 1, 1)
            # -------------------- #
            self.upsample13 = torch.nn.Upsample(scale_factor=2)  # input_size/16
            self.l8_add = cbs(config[18 + 6 * n], config[33 + 8 * n], 1, 1)
            self.concat14 = concat(dim=1)
            self.elan_h15 = elan_h(config[32 + 8 * n] + config[33 + 8 * n], None, config[34 + 8 * n:41 + 8 * n])
            self.cbs16 = cbs(config[40 + 8 * n], config[41 + 8 * n], 1, 1)
            # -------------------- #
            self.upsample17 = torch.nn.Upsample(scale_factor=2)  # input_size/8
            self.l6_add = cbs(config[12 + 4 * n], config[42 + 8 * n], 1, 1)
            self.concat18 = concat(dim=1)
            self.elan_h19 = elan_h(config[41 + 8 * n] + config[42 + 8 * n], None,
                                   config[43 + 8 * n:50 + 8 * n])  # 接output0
            # -------------------- #
            self.mp20 = mp(config[49 + 8 * n], None, config[50 + 8 * n:53 + 8 * n])
            self.concat21 = concat(dim=1)
            self.elan_h22 = elan_h(config[40 + 8 * n] + config[50 + 8 * n] + config[52 + 8 * n], None,
                                   config[53 + 8 * n:60 + 8 * n])  # 接output1
            # -------------------- #
            self.mp23 = mp(config[59 + 8 * n], None, config[60 + 8 * n:63 + 8 * n])
            self.concat24 = concat(dim=1)
            self.elan_h25 = elan_h(config[31 + 8 * n] + config[60 + 8 * n] + config[62 + 8 * n], None,
                                   config[63 + 8 * n:70 + 8 * n])  # 接output2
            # -------------------- #
            self.head0 = head(config[49 + 8 * n], self.output_size[0], self.output_class)
            self.head1 = head(config[59 + 8 * n], self.output_size[1], self.output_class)
            self.head2 = head(config[69 + 8 * n], self.output_size[2], self.output_class)

    def forward(self, x):
        x = self.cbs0(x)
        x = self.cbs1(x)
        x = self.cbs2(x)
        x = self.cbs3(x)
        x = self.elan4(x)
        x = self.mp5(x)
        l6 = self.elan6(x)
        x = self.mp7(l6)
        l8 = self.elan8(x)
        x = self.mp9(l8)
        x = self.elan10(x)
        l11 = self.sppcspc11(x)
        x = self.cbs12(l11)
        x = self.upsample13(x)
        l8_add = self.l8_add(l8)
        x = self.concat14([x, l8_add])
        l15 = self.elan_h15(x)
        x = self.cbs16(l15)
        x = self.upsample17(x)
        l6_add = self.l6_add(l6)
        x = self.concat18([x, l6_add])
        x = self.elan_h19(x)
        head0 = self.head0(x)
        x = self.mp20(x)
        x = self.concat21([x, l15])
        x = self.elan_h22(x)
        head1 = self.head1(x)
        x = self.mp23(x)
        x = self.concat24([x, l11])
        x = self.elan_h25(x)
        head2 = self.head2(x)
        x = self.decode([head0, head1, head2])
        return x


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_type', default='s', type=str)
    parser.add_argument('--input_size', default=640, type=int)
    parser.add_argument('--output_class', default=1, type=int)
    parser.add_argument('--output_size', default=(80, 40, 20), type=tuple)
    parser.add_argument('--output_layer', default=(3, 3, 3), type=tuple)
    parser.add_argument('--anchor', default=(((0.02, 0.03), (0.03, 0.06), (0.06, 0.04)),
                                             ((0.06, 0.11), (0.11, 0.08), (0.11, 0.22)),
                                             ((0.22, 0.17), (0.30, 0.40), (0.72, 0.62))))
    args = parser.parse_args()
    model = yolov7(args)
    tensor = torch.rand(2, 3, args.input_size, args.input_size, dtype=torch.float32)
    pred = model(tensor)
    print(model)
    print(pred.shape)
