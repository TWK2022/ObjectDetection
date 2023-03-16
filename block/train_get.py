import cv2
import tqdm
import wandb
import torch
import numpy as np
from block.val_get import val_get


def train_get(args, data_dict, model_dict, loss):
    model = model_dict['model'].to(args.device, non_blocking=args.latch)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_dataloader = torch.utils.data.DataLoader(torch_dataset(args, 'train', data_dict['train'], data_dict['class']),
                                                   batch_size=args.batch, shuffle=True, drop_last=True,
                                                   pin_memory=args.latch, num_workers=args.num_worker)
    val_dataloader = torch.utils.data.DataLoader(torch_dataset(args, 'val', data_dict['val'], data_dict['class']),
                                                 batch_size=args.batch, shuffle=False, drop_last=False,
                                                 pin_memory=args.latch, num_workers=args.num_worker)
    for epoch in range(args.epoch):
        # 训练
        print(f'\n-----------------------第{epoch + 1}轮-----------------------')
        model.train()
        train_loss = 0  # 记录训练损失
        train_frame_loss = 0  # 记录边框损失
        train_confidence_loss = 0  # 记录置信度框损失
        train_class_loss = 0  # 记录类别损失
        for item, (image_batch, true_batch, judge_batch) in enumerate(tqdm.tqdm(train_dataloader)):
            image_batch = image_batch.to(args.device, non_blocking=args.latch)  # 将输入数据放到设备上
            for i in range(len(true_batch)):  # 将标签矩阵放到对应设备上
                true_batch[i] = true_batch[i].to(args.device, non_blocking=args.latch)
            pred_batch = model(image_batch)
            # 计算损失
            loss_batch, frame_loss, confidence_loss, class_loss = loss(pred_batch, true_batch, judge_batch)
            train_loss += loss_batch.item()
            train_frame_loss += frame_loss.item()
            train_confidence_loss += confidence_loss.item()
            train_class_loss += class_loss.item()
            # 更新参数
            optimizer.zero_grad()
            if args.scaler:
                args.scaler.scale(loss_batch).backward()
                args.scaler.step(optimizer)
                args.scaler.update()
            else:
                loss_batch.backward()
                optimizer.step()
        # 计算平均损失
        train_loss = train_loss / (item + 1)
        train_frame_loss = train_frame_loss / (item + 1)
        train_confidence_loss = train_confidence_loss / (item + 1)
        train_class_loss = train_class_loss / (item + 1)
        print('\n| 轮次:{} | train_loss:{:.4f} | train_frame_loss:{:.4f} | train_confidence_loss:{:.4f} |'
              ' train_class_loss:{:.4f} |\n'
              .format(item + 1, train_loss, train_frame_loss, train_confidence_loss, train_class_loss))
        # 清理显存空间
        del image_batch, true_batch, pred_batch, judge_batch, loss_batch
        torch.cuda.empty_cache()
        # 验证
        val_loss, val_frame_loss, val_confidence_loss, val_class_loss, accuracy, precision, recall, m_ap = \
            val_get(args, val_dataloader, model, loss)
        # 保存
        if m_ap > 0.25 and m_ap > model_dict['val_m_ap']:
            model_dict['model'] = model
            model_dict['class'] = data_dict['class']
            model_dict['epoch'] = epoch
            model_dict['train_loss'] = train_loss
            model_dict['train_frame_loss'] = train_frame_loss
            model_dict['train_confidence_loss'] = train_confidence_loss
            model_dict['train_class_loss'] = train_class_loss
            model_dict['val_loss'] = val_loss
            model_dict['val_frame_loss'] = val_frame_loss
            model_dict['val_confidence_loss'] = val_confidence_loss
            model_dict['val_class_loss'] = val_class_loss
            model_dict['val_accuracy'] = accuracy
            model_dict['val_precision'] = precision
            model_dict['val_recall'] = recall
            model_dict['val_m_ap'] = m_ap
            torch.save(model_dict, args.save_name)
            print('\n| 保存模型:{} | val_m_ap:{:.4f} |\n'
                  .format(args.save_name, m_ap))
        # wandb
        if args.wandb:
            args.wandb_run.log({'train/train_loss': train_loss,
                                'train/train_frame_loss': train_frame_loss,
                                'train/train_confidence_loss': train_confidence_loss,
                                'train/train_class_loss': train_class_loss,
                                'val_loss/val_loss': val_loss,
                                'val_loss/val_frame_loss': val_frame_loss,
                                'val_loss/val_confidence_loss': val_confidence_loss,
                                'val_loss/val_class_loss': val_class_loss,
                                'val_metric/val_accuracy': accuracy,
                                'val_metric/val_precision': precision,
                                'val_metric/val_recall': recall,
                                'val_metric/val_m_ap': m_ap})
    return model_dict


class torch_dataset(torch.utils.data.Dataset):
    def __init__(self, args, tag, data, class_name):
        output_num_dict = {'yolov5': (3, 3, 3),
                           'yolov7': (3, 3, 3)
                           }  # 输出层数量，如(3, 3, 3)代表有三个大层，每层有三个小层
        stride_dict = {'yolov5': (8, 16, 32),
                       'yolov7': (8, 16, 32)
                       }  # 每个输出层尺寸缩小的幅度
        anchor_dict = {'yolov5': (((10, 13), (16, 30), (33, 23)), ((30, 61), (62, 45), (59, 119)),
                                  ((116, 90), (156, 198), (373, 326))),
                       'yolov7': (((10, 13), (16, 30), (33, 23)), ((30, 61), (62, 45), (59, 119)),
                                  ((116, 90), (156, 198), (373, 326)))}  # 先验框
        wh_multiple_dict = {'yolov5': 4,
                            'yolov7': 4
                            }  # 宽高的倍数，真实wh=网络原始输出[0-1]*倍数*anchor
        self.output_num = output_num_dict[args.model]
        self.stride = stride_dict[args.model]
        self.anchor = anchor_dict[args.model]
        self.wh_multiple = wh_multiple_dict[args.model]
        self.input_size = args.input_size  # 输入尺寸，如640
        self.output_class = args.output_class  # 输出类别数
        self.label_smooth = args.label_smooth  # 标签平滑，如(0.05,0.95)
        self.output_size = [int(self.input_size // i) for i in self.stride]  # 每个输出层的尺寸，如(80,40,20)
        self.tag = tag  # 用于区分是训练集还是验证集
        self.data = data
        # wandb可视化部分
        self.wandb = args.wandb
        if self.wandb:
            self.class_name = class_name
            self.wandb_run = args.wandb_run
            self.wandb_num = 0  # 用于限制添加的图片数量(最多添加20张)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 图片和标签处理，边框坐标处理为真实的Cx,Cy,w,h
        image = cv2.imread(self.data[index][0])  # 读取图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为RGB通道
        label = self.data[index][1]  # 读取原始标签([:,类别号+Cx,Cy,w,h]，边框为相对边长的比例值)
        image, frame = self._resize(image, label[:, 1:])  # 缩放和填充图片(归一化、减均值、除以方差、调维度等在模型中完成)
        image = torch.tensor(image, dtype=torch.float32)
        # 边框:将相对坐标变为绝对坐标
        frame = torch.tensor(frame, dtype=torch.float32) * self.input_size
        # 置信度:为1
        confidence = torch.ones((len(label), 1), dtype=torch.float32)
        # 类别:类别独热编码
        cls = torch.full((len(label), self.output_class), self.label_smooth[0], dtype=torch.float32)
        for i in range(len(label)):
            cls[i][int(label[i, 0])] = self.label_smooth[1]
        # 合并为标签
        label = torch.concat([frame, confidence, cls], dim=1)
        # 标签矩阵处理
        label_list = [0 for _ in range(len(self.output_num))]  # 存放每个输出层的标签矩阵
        judge_list = [0 for _ in range(len(self.output_num))]  # 存放每个输出层的判断矩阵
        for i in range(len(self.output_num)):  # 遍历每个输出层
            label_matrix = torch.zeros(self.output_num[i], self.output_size[i], self.output_size[i],
                                       5 + self.output_class, dtype=torch.float32)  # 标签矩阵
            judge_matrix = torch.zeros(self.output_num[i], self.output_size[i], self.output_size[i],
                                       dtype=torch.bool)  # 判断矩阵，False代表没有标签
            # 标签对应输出网格的坐标
            Cx = frame[:, 0] / self.stride[i]
            x_grid = Cx.type(torch.int8)
            x_move = Cx - x_grid
            x_grid_add = x_grid + 2 * torch.round(x_move).type(torch.int8) - 1  # 每个标签可以由相邻网格预测
            x_grid_add = torch.clamp(x_grid_add, 0, self.output_size[i] - 1)  # 网格不能超出范围(与x_grid重复的网格之后不会加入)
            Cy = frame[:, 1] / self.stride[i]
            y_grid = Cy.type(torch.int8)
            y_move = Cy - y_grid
            y_grid_add = y_grid + 2 * torch.round(y_move).type(torch.int8) - 1  # 每个标签可以由相邻网格预测
            y_grid_add = torch.clamp(y_grid_add, 0, self.output_size[i] - 1)  # 网格不能超出范围(与y_grid重复的网格之后不会加入)
            # 遍历每个输出层的小层
            for j in range(self.output_num[i]):
                # 根据wh筛选
                w = frame[:, 2] / self.anchor[i][j][0] / self.wh_multiple  # 该值要在0-1该层才能预测(但0-0.0625太小可以舍弃)
                h = frame[:, 3] / self.anchor[i][j][1] / self.wh_multiple  # 该值要在0-1该层才能预测(但0-0.0625太小可以舍弃)
                wh_screen = torch.where((0.0625 < w) & (w < 1) & (0.0625 < h) & (h < 1), True, False)  # 筛选可以预测的标签
                # 将标签填入对应的标签矩阵位置
                for k in range(len(label)):
                    if wh_screen[k]:
                        label_matrix[j, x_grid[k], y_grid[k]] = label[k]
                        judge_matrix[j, x_grid[k], y_grid[k]] = True
                # 将扩充的标签填入对应的标签矩阵位置
                for k in range(len(label)):
                    if wh_screen[k] and not judge_matrix[j, x_grid_add[k], y_grid[k]]:
                        label_matrix[j, x_grid_add[k], y_grid[k]] = label[k]
                        judge_matrix[j, x_grid_add[k], y_grid[k]] = True
                    if wh_screen[k] and not judge_matrix[j, x_grid[k], y_grid_add[k]]:
                        label_matrix[j, x_grid[k], y_grid_add[k]] = label[k]
                        judge_matrix[j, x_grid[k], y_grid_add[k]] = True
            # 存放每个输出层的结果
            label_list[i] = label_matrix
            judge_list[i] = judge_matrix
        # 使用wandb添加图片
        if self.wandb and self.wandb_num < 20:
            box_data = []
            cls_num = torch.argmax(cls, dim=1)
            frame[:, 0:2] = frame[:, 0:2] - 1 / 2 * frame[:, 2:4]
            frame[:, 2:4] = frame[:, 0:2] + frame[:, 2:4]
            for i in range(len(frame)):
                class_id = cls_num[i]
                box_data.append({"position": {"minX": frame[i][0].item() / self.input_size,
                                              "minY": frame[i][1].item() / self.input_size,
                                              "maxX": frame[i][2].item() / self.input_size,
                                              "maxY": frame[i][3].item() / self.input_size},
                                 "class_id": class_id.item(), "box_caption": self.class_name[class_id]})
            wandb_image = wandb.Image(np.array(image, dtype=np.uint8), boxes={"predictions": {"box_data": box_data}})
            self.wandb_run.log({f'image/{self.tag}_image': wandb_image})
            self.wandb_num += 1
        return image, label_list, judge_list

    def _resize(self, image, frame):  # 将图片四周填充变为正方形，frame输入输出都为[[Cx,Cy,w,h]...](相对原图片的比例值)
        shape = image.shape
        w0 = shape[1]
        h0 = shape[0]
        if w0 == h0 == self.input_size:  # 不需要变形
            return image, frame
        else:
            image_resize = np.full((self.input_size, self.input_size, 3), 127)
            if w0 >= h0:  # 宽大于高
                w = self.input_size
                h = int(w / w0 * h0)
                image = cv2.resize(image, (w, h))
                add_y = (w - h) // 2
                image_resize[add_y:add_y + h] = image
                frame[:, 0] = np.around(frame[:, 0] * w)
                frame[:, 1] = np.around(frame[:, 1] * h + add_y)
                frame[:, 2] = np.around(frame[:, 2] * w)
                frame[:, 3] = np.around(frame[:, 3] * h)
                return image_resize, frame
            else:  # 宽小于高
                h = self.input_size
                w = int(h / h0 * w0)
                image = cv2.resize(image, (w, h))
                add_x = (h - w) // 2
                image_resize[:, add_x:add_x + w] = image
                frame[:, 0] = np.around(frame[:, 0] * w + add_x)
                frame[:, 1] = np.around(frame[:, 1] * h)
                frame[:, 2] = np.around(frame[:, 2] * w)
                frame[:, 3] = np.around(frame[:, 3] * h)
                return image_resize, frame
