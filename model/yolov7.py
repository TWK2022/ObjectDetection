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
            self.l8_concat = cbs(32 * dim, 8 * dim, 1, 1)
            self.concat14 = concat(1)
            self.elan_h15 = elan_h(16 * dim, 8 * dim)
            self.cbs16 = cbs(8 * dim, 4 * dim, 1, 1)
            # -------------------- #
            self.upsample17 = torch.nn.Upsample(scale_factor=2)  # input_size/8
            self.l6_concat = cbs(16 * dim, 4 * dim, 1, 1)
            self.concat18 = concat(1)
            self.elan_h19 = elan_h(8 * dim, 4 * dim)  # 接head0
            # -------------------- #
            self.mp20 = mp(4 * dim, 8 * dim)
            self.concat21 = concat(1)
            self.elan_h22 = elan_h(16 * dim, 8 * dim)  # 接head1
            # -------------------- #
            self.mp23 = mp(8 * dim, 16 * dim)
            self.concat24 = concat(1)
            self.elan_h25 = elan_h(32 * dim, 16 * dim)  # 接head2
            # -------------------- #
            self.head0 = head(4 * dim, self.output_size[0], self.output_class)
            self.head1 = head(8 * dim, self.output_size[1], self.output_class)
            self.head2 = head(16 * dim, self.output_size[2], self.output_class)
        else:  # 剪枝版本
            self.cbs0 = cbs(3, config[0], 3, 1)
            self.cbs1 = cbs(config[0], config[1], 3, 2)  # input_size/2
            self.cbs2 = cbs(config[1], config[2], 3, 1)
            self.cbs3 = cbs(config[2], config[3], 3, 2)  # input_size/4
            index = 4
            self.elan4 = elan(config[index - 1], n=n, config=config[index:])
            index += self.elan4.config_len
            self.mp5 = mp(self.elan4.last_layer, config=config[index:])
            index += self.mp5.config_len
            self.elan6 = elan(self.mp5.last_layer, n=n, config=config[index:])
            index += self.elan6.config_len
            self.mp7 = mp(self.elan6.last_layer, config=config[index:])
            index += self.mp7.config_len
            self.elan8 = elan(self.mp7.last_layer, n=n, config=config[index:])
            index += self.elan8.config_len
            self.mp9 = mp(self.elan8.last_layer, config=config[index:])
            index += self.mp9.config_len
            self.elan10 = elan(self.mp9.last_layer, n=n, config=config[index:])
            index += self.elan10.config_len
            self.sppcspc11 = sppcspc(self.elan10.last_layer, config=config[index:])
            index += self.sppcspc11.config_len
            self.cbs12 = cbs(config[index - 1], config[index], 1, 1)
            index += 1
            # -------------------- #
            self.upsample13 = torch.nn.Upsample(scale_factor=2)
            self.l8_concat = cbs(self.elan8.last_layer, config[index], 1, 1)
            index += 1
            self.concat14 = concat(dim=1)
            self.elan_h15 = elan_h(config[index - 1] + config[index - 2],
                                   config=config[index:])
            index += self.elan_h15.config_len
            self.cbs16 = cbs(config[index - 1], config[index], 1, 1)
            index += 1
            # -------------------- #
            self.upsample17 = torch.nn.Upsample(scale_factor=2)
            self.l6_concat = cbs(self.elan6.last_layer, config[index], 1, 1)
            index += 1
            self.concat18 = concat(dim=1)
            self.elan_h19 = elan_h(config[index - 1] + config[index - 2], config=config[index:])
            index += self.elan_h19.config_len
            # -------------------- #
            self.mp20 = mp(self.elan_h19.last_layer, config=config[index:])
            index += self.mp20.config_len
            self.concat21 = concat(dim=1)
            self.elan_h22 = elan_h(self.mp20.last_layer + self.elan_h15.last_layer,
                                   config=config[index:])
            index += self.elan_h22.config_len
            # -------------------- #
            self.mp23 = mp(self.elan_h22.last_layer, config=config[index:])
            index += self.mp23.config_len
            self.concat24 = concat(dim=1)
            self.elan_h25 = elan_h(self.mp23.last_layer + self.sppcspc11.last_layer, config=config[index:])
            index += self.elan_h25.config_len
            # -------------------- #
            self.head0 = head(self.elan_h19.last_layer, self.output_size[0], self.output_class)
            self.head1 = head(self.elan_h22.last_layer, self.output_size[1], self.output_class)
            self.head2 = head(self.elan_h25.last_layer, self.output_size[2], self.output_class)

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
        l8_concat = self.l8_concat(l8)
        x = self.concat14([x, l8_concat])
        l15 = self.elan_h15(x)
        x = self.cbs16(l15)
        x = self.upsample17(x)
        l6_concat = self.l6_concat(l6)
        x = self.concat18([x, l6_concat])
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
