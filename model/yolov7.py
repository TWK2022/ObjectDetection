# 根据yolov7改编:https://github.com/WongKinYiu/yolov7
import torch
from model.layer import cbs, elan, elan_h, mp, sppcspc, concat, head


class yolov7(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        input_size = args.input_size
        stride = (8, 16, 32)
        self.output_size = [int(input_size // i) for i in stride]  # 每个输出层的尺寸，如(80,40,20)
        self.output_class = args.output_class
        dim_dict = {'n': 8, 's': 16, 'm': 32, 'l': 64}
        n_dict = {'n': 1, 's': 1, 'm': 2, 'l': 3}
        dim = dim_dict[args.model_type]
        n = n_dict[args.model_type]
        # 网络结构
        if not args.prune:  # 正常版本
            self.l0 = cbs(3, dim, 3, 1)
            self.l1 = cbs(dim, 2 * dim, 3, 2)  # input_size/2
            self.l2 = cbs(2 * dim, 2 * dim, 3, 1)
            self.l3 = cbs(2 * dim, 4 * dim, 3, 2)  # input_size/4
            # ---------- #
            self.l4 = elan(4 * dim, 8 * dim, n)
            self.l5 = mp(8 * dim, 8 * dim)  # input_size/8
            self.l6 = elan(8 * dim, 16 * dim, n)
            self.l7 = mp(16 * dim, 16 * dim)  # input_size/16
            self.l8 = elan(16 * dim, 32 * dim, n)
            self.l9 = mp(32 * dim, 32 * dim)  # input_size/32
            self.l10 = elan(32 * dim, 32 * dim, n)
            self.l11 = sppcspc(32 * dim, 16 * dim)
            self.l12 = cbs(16 * dim, 8 * dim, 1, 1)
            # ---------- #
            self.l13 = torch.nn.Upsample(scale_factor=2)  # input_size/16
            self.l8_add = cbs(32 * dim, 8 * dim, 1, 1)
            self.l14 = concat(1)
            self.l15 = elan_h(16 * dim, 8 * dim)
            self.l16 = cbs(8 * dim, 4 * dim, 1, 1)
            # ---------- #
            self.l17 = torch.nn.Upsample(scale_factor=2)  # input_size/8
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
            self.output0 = head(4 * dim, self.output_size[0], self.output_class)
            self.output1 = head(8 * dim, self.output_size[1], self.output_class)
            self.output2 = head(16 * dim, self.output_size[2], self.output_class)
        else:  # 剪枝版本
            config = args.prune_num
            self.l0 = cbs(3, config[0], 1, 1)
            self.l1 = cbs(config[0], config[1], 3, 2)  # input_size/2
            self.l2 = cbs(config[1], config[2], 1, 1)
            self.l3 = cbs(config[2], config[3], 3, 2)  # input_size/4
            # ---------- #
            self.l4 = elan(config[3], None, n, config[4:7 + 2 * n])
            self.l5 = mp(config[6 + 2 * n], None, config[7 + 2 * n:10 + 2 * n])  # input_size/8
            self.l6 = elan(config[7 + 2 * n] + config[9 + 2 * n], None, n, config[10 + 2 * n:13 + 4 * n])
            self.l7 = mp(config[12 + 4 * n], None, config[13 + 4 * n:16 + 4 * n])  # input_size/16
            self.l8 = elan(config[13 + 4 * n] + config[15 + 4 * n], None, n, config[16 + 4 * n:19 + 6 * n])
            self.l9 = mp(config[18 + 6 * n], None, config[19 + 6 * n:22 + 6 * n])  # input_size/32
            self.l10 = elan(config[19 + 6 * n] + config[21 + 6 * n], None, n, config[22 + 6 * n:25 + 8 * n])
            self.l11 = sppcspc(config[24 + 8 * n], None, config[25 + 8 * n:32 + 8 * n])
            self.l12 = cbs(config[31 + 8 * n], config[32 + 8 * n], 1, 1)
            # ---------- #
            self.l13 = torch.nn.Upsample(scale_factor=2)  # input_size/16
            self.l8_add = cbs(config[18 + 6 * n], config[33 + 8 * n], 1, 1)
            self.l14 = concat(1)
            self.l15 = elan_h(config[32 + 8 * n] + config[33 + 8 * n], None, config[34 + 8 * n:41 + 8 * n])
            self.l16 = cbs(config[40 + 8 * n], config[41 + 8 * n], 1, 1)
            # ---------- #
            self.l17 = torch.nn.Upsample(scale_factor=2)  # input_size/8
            self.l6_add = cbs(config[12 + 4 * n], config[42 + 8 * n], 1, 1)
            self.l18 = concat(1)
            self.l19 = elan_h(config[41 + 8 * n] + config[42 + 8 * n], None, config[43 + 8 * n:50 + 8 * n])  # 接output0
            # ---------- #
            self.l20 = mp(config[49 + 8 * n], None, config[50 + 8 * n:53 + 8 * n])
            self.l21 = concat(1)
            self.l22 = elan_h(config[40 + 8 * n] + config[50 + 8 * n] + config[52 + 8 * n], None,
                              config[53 + 8 * n:60 + 8 * n])  # 接output1
            # ---------- #
            self.l23 = mp(config[59 + 8 * n], None, config[60 + 8 * n:63 + 8 * n])
            self.l24 = concat(1)
            self.l25 = elan_h(config[31 + 8 * n] + config[60 + 8 * n] + config[62 + 8 * n], None,
                              config[63 + 8 * n:70 + 8 * n])  # 接output2
            # ---------- #
            self.output0 = head(config[49 + 8 * n], self.output_size[0], self.output_class)
            self.output1 = head(config[59 + 8 * n], self.output_size[1], self.output_class)
            self.output2 = head(config[69 + 8 * n], self.output_size[2], self.output_class)

    def forward(self, x):
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
        x = self.l20(x)
        x = self.l21([x, l15])
        x = self.l22(x)
        output1 = self.output1(x)
        x = self.l23(x)
        x = self.l24([x, l11])
        x = self.l25(x)
        output2 = self.output2(x)
        return [output0, output1, output2]


if __name__ == '__main__':
    import argparse
    from layer import cbs, elan, elan_h, mp, sppcspc, concat, head

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--prune', default=False, type=bool)
    parser.add_argument('--model_type', default='n', type=str)
    parser.add_argument('--input_size', default=640, type=int)
    parser.add_argument('--output_class', default=1, type=int)
    args = parser.parse_args()
    model = yolov7(args).to('cpu')
    tensor = torch.rand(2, 3, args.input_size, args.input_size, dtype=torch.float32).to('cpu')
    pred = model(tensor)
    print(model)
    print(pred[0].shape, pred[1].shape, pred[2].shape)
