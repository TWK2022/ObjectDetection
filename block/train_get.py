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
                                                   pin_memory=args.latch, num_workers=args.num_worker,
                                                   collate_fn=collate_fn)
    val_dataloader = torch.utils.data.DataLoader(torch_dataset(args, 'val', data_dict['val'], data_dict['class']),
                                                 batch_size=args.batch, shuffle=False, drop_last=False,
                                                 pin_memory=args.latch, num_workers=args.num_worker,
                                                 collate_fn=collate_fn)
    for epoch in range(args.epoch):
        # 训练
        print(f'\n-----------------------第{epoch + 1}轮-----------------------')
        model.train()
        train_loss = 0  # 记录训练损失
        train_frame_loss = 0  # 记录边框损失
        train_confidence_loss = 0  # 记录置信度框损失
        train_class_loss = 0  # 记录类别损失
        for item, (image_batch, true_batch, judge_batch, label_list) in enumerate(tqdm.tqdm(train_dataloader)):
            image_batch = image_batch.to(args.device, non_blocking=args.latch)  # 将输入数据放到设备上
            for i in range(len(true_batch)):  # 将标签矩阵放到对应设备上
                true_batch[i] = true_batch[i].to(args.device, non_blocking=args.latch)
            if args.scaler:
                with torch.cuda.amp.autocast():
                    pred_batch = model(image_batch)
                    loss_batch, frame_loss, confidence_loss, class_loss = loss(pred_batch, true_batch, judge_batch)
                optimizer.zero_grad()
                args.scaler.scale(loss_batch).backward()
                args.scaler.step(optimizer)
                args.scaler.update()
            else:
                pred_batch = model(image_batch)
                loss_batch, frame_loss, confidence_loss, class_loss = loss(pred_batch, true_batch, judge_batch)
                optimizer.zero_grad()
                loss_batch.backward()
                optimizer.step()
            # 记录损失
            train_loss += loss_batch.item()
            train_frame_loss += frame_loss.item()
            train_confidence_loss += confidence_loss.item()
            train_class_loss += class_loss.item()
        # 计算平均损失
        train_loss = train_loss / (item + 1)
        train_frame_loss = train_frame_loss / (item + 1)
        train_confidence_loss = train_confidence_loss / (item + 1)
        train_class_loss = train_class_loss / (item + 1)
        print('\n| 轮次:{} | train_loss:{:.4f} | train_frame_loss:{:.4f} | train_confidence_loss:{:.4f} |'
              ' train_class_loss:{:.4f} |\n'
              .format(item + 1, train_loss, train_frame_loss, train_confidence_loss, train_class_loss))
        # 清理显存空间
        del image_batch, true_batch, judge_batch, pred_batch, loss_batch
        torch.cuda.empty_cache()
        # 验证
        val_loss, val_frame_loss, val_confidence_loss, val_class_loss, accuracy, precision, recall, m_ap, \
        nms_precision, nms_recall, nms_m_ap = val_get(args, val_dataloader, model, loss)
        # 保存
        model_dict['model'] = model
        model_dict['class'] = data_dict['class']
        model_dict['epoch'] = epoch
        model_dict['train_loss'] = train_loss
        model_dict['val_loss'] = val_loss
        model_dict['val_m_ap'] = m_ap
        model_dict['val_nms_m_ap'] = nms_m_ap
        torch.save(model_dict, 'last.pt')  # 保存最后一次训练的模型
        if m_ap > 0.25 and m_ap > model_dict['val_m_ap']:
            torch.save(model_dict, args.save_name)  # 保存最佳模型
            print('\n| 保存最佳模型:{} | val_m_ap:{:.4f} |\n'.format(args.save_name, m_ap))
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
                                'val_metric/val_m_ap': m_ap,
                                'val_nms_metric/val_nms_precision': nms_precision,
                                'val_nms_metric/val_nms_recall': nms_recall,
                                'val_nms_metric/val_nms_m_ap': nms_m_ap})
    return model_dict


class torch_dataset(torch.utils.data.Dataset):
    def __init__(self, args, tag, data, class_name):
        output_num_dict = {'yolov5': (3, 3, 3),
                           'yolov7': (3, 3, 3)
                           }  # 输出层数量，如(3, 3, 3)代表有三个大层，每层有三个小层
        stride_dict = {'yolov5': (8, 16, 32),
                       'yolov7': (8, 16, 32)
                       }  # 每个输出层尺寸缩小的幅度
        anchor_dict = {'yolov5': (((12, 16), (19, 36), (40, 28)), ((36, 75), (76, 55), (72, 146)),
                                  ((142, 110), (192, 243), (459, 401))),
                       'yolov7': (((12, 16), (19, 36), (40, 28)), ((36, 75), (76, 55), (72, 146)),
                                  ((142, 110), (192, 243), (459, 401)))
                       }  # 先验框
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
        self.mosaic = args.mosaic
        # wandb可视化部分
        self.wandb = args.wandb
        if self.wandb:
            self.wandb_run = args.wandb_run
            self.wandb_count = 0  # 用于限制添加的图片数量(最多添加args.wandb_image_num张)
            self.wandb_image_num = args.wandb_image_num
            self.wandb_image = []  # 记录所有的image最后一起添加
            self.wandb_class_name = {}  # 用于给边框添加标签名字
            for i in range(len(class_name)):
                self.wandb_class_name[i] = class_name[i]

    def __len__(self):
        return len(self.data)

    def draw(self, image, frame):  # 输入(x_min,y_min,w,h)真实坐标
        image = image.astype(np.uint8)
        for i in range(len(frame)):
            a = (int(frame[i][0]), int(frame[i][1]))
            b = (int(frame[i][0] + frame[i][2]), int(frame[i][1] + frame[i][3]))
            cv2.rectangle(image, a, b, color=(0, 255, 0), thickness=2)
        cv2.imshow('pred', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __getitem__(self, index):
        # 图片和标签处理，边框坐标处理为真实的Cx,Cy,w,h(归一化、减均值、除以方差、调维度等在模型中完成)
        if self.tag == 'train' and torch.rand(1) < self.mosaic:
            index_mix = torch.randperm(len(self.data))[0:4]
            index_mix[0] = index
            image, frame, label = self._mosaic(index_mix)  # 马赛克增强，相对坐标(Cx,Cy,w,h)变为真实坐标
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)  # 转为RGB通道
        else:
            image = cv2.imread(self.data[index][0])  # 读取图片
            label = self.data[index][1].copy()  # 读取原始标签([:,类别号+Cx,Cy,w,h]，边框为相对边长的比例值)
            image, frame = self._resize(image.astype(np.uint8), label[:, 1:],
                                        self.input_size)  # 缩放和填充图片，相对坐标(Cx,Cy,w,h)变为真实坐标
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)  # 转为RGB通道
        image = torch.tensor(image, dtype=torch.float32)
        # 边框:转换为张量
        frame = torch.tensor(frame, dtype=torch.float32)
        # 置信度:为1
        confidence = torch.ones((len(label), 1), dtype=torch.float32)
        # 类别:类别独热编码
        cls = torch.full((len(label), self.output_class), self.label_smooth[0], dtype=torch.float32)
        for i in range(len(label)):
            cls[i][int(label[i, 0])] = self.label_smooth[1]
        # 合并为标签
        label = torch.concat([frame, confidence, cls], dim=1)
        # 标签矩阵处理
        label_matrix_list = [0 for _ in range(len(self.output_num))]  # 存放每个输出层的标签矩阵
        judge_matrix_list = [0 for _ in range(len(self.output_num))]  # 存放每个输出层的判断矩阵
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
            label_matrix_list[i] = label_matrix
            judge_matrix_list[i] = judge_matrix
        # 使用wandb添加图片
        if self.wandb and self.wandb_count < self.wandb_image_num:
            self.wandb_count += 1
            box_data = []
            cls_num = torch.argmax(cls, dim=1)
            frame[:, 0:2] = frame[:, 0:2] - frame[:, 2:4] / 2
            frame[:, 2:4] = frame[:, 0:2] + frame[:, 2:4]
            frame = frame / self.input_size  # (x_min,y_min,x_max,y_max)相对坐标
            for i in range(len(frame)):
                class_id = cls_num[i].item()
                box_data.append({"position": {"minX": frame[i][0].item(),
                                              "minY": frame[i][1].item(),
                                              "maxX": frame[i][2].item(),
                                              "maxY": frame[i][3].item()},
                                 "class_id": class_id,
                                 "box_caption": self.wandb_class_name[class_id]})
            wandb_image = wandb.Image(np.array(image, dtype=np.uint8),
                                      boxes={"predictions": {"box_data": box_data,
                                                             'class_labels': self.wandb_class_name}})
            self.wandb_image.append(wandb_image)
            if self.wandb_count == self.wandb_image_num:
                self.wandb_run.log({f'image/{self.tag}_image': self.wandb_image})
        return image, label_matrix_list, judge_matrix_list, label

    def _resize(self, image, frame, input_size):  # 将图片四周填充变为正方形，frame输入输出都为[[Cx,Cy,w,h]...](相对原图片的比例值)
        shape = image.shape
        w0 = shape[1]
        h0 = shape[0]
        if w0 == h0 == input_size:  # 不需要变形
            frame *= input_size
            return image, frame
        else:
            image_resize = np.full((input_size, input_size, 3), 127)
            if w0 >= h0:  # 宽大于高
                w = input_size
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
                h = input_size
                w = int(h / h0 * w0)
                image = cv2.resize(image, (w, h))
                add_x = (h - w) // 2
                image_resize[:, add_x:add_x + w] = image
                frame[:, 0] = np.around(frame[:, 0] * w + add_x)
                frame[:, 1] = np.around(frame[:, 1] * h)
                frame[:, 2] = np.around(frame[:, 2] * w)
                frame[:, 3] = np.around(frame[:, 3] * h)
                return image_resize, frame

    def _mosaic(self, index_mix):
        image0 = cv2.imread(self.data[index_mix[0]][0])
        image1 = cv2.imread(self.data[index_mix[1]][0])
        image2 = cv2.imread(self.data[index_mix[2]][0])
        image3 = cv2.imread(self.data[index_mix[3]][0])
        label0 = self.data[index_mix[0]][1].copy()
        label1 = self.data[index_mix[1]][1].copy()
        label2 = self.data[index_mix[2]][1].copy()
        label3 = self.data[index_mix[3]][1].copy()
        image_resize = np.full((self.input_size, self.input_size, 3), 127)
        image0, frame0 = self._resize(image0, label0[:, 1:], self.input_size // 2)
        image1, frame1 = self._resize(image1, label1[:, 1:], self.input_size // 2)
        image2, frame2 = self._resize(image2, label2[:, 1:], self.input_size // 2)
        image3, frame3 = self._resize(image3, label3[:, 1:], self.input_size // 2)
        image_resize[0:self.input_size // 2, 0:self.input_size // 2] = image0
        image_resize[self.input_size // 2:self.input_size, 0:self.input_size // 2] = image1
        image_resize[0:self.input_size // 2, self.input_size // 2:self.input_size] = image2
        image_resize[self.input_size // 2:self.input_size, self.input_size // 2:self.input_size] = image3
        frame1[:, 1] += self.input_size // 2
        frame2[:, 0] += self.input_size // 2
        frame3[:, 0:2] += self.input_size // 2
        frame = np.concatenate([frame0, frame1, frame2, frame3], axis=0)
        label = np.concatenate([label0, label1, label2, label3], axis=0)
        return image_resize, frame, label


def collate_fn(batch):  # 自定义__getitem__合并方式
    image_list = []
    label_matrix_list = [[] for _ in range(len(batch[0][1]))]
    judge_matrix_list = [[] for _ in range(len(batch[0][2]))]
    label_list = []
    for i in range(len(batch)):  # 遍历所有__getitem__
        image = batch[i][0]
        label_matrix = batch[i][1]
        judge_matrix = batch[i][2]
        label = batch[i][3]
        image_list.append(image)
        for j in range(len(label_matrix)):  # 遍历每个输出层
            label_matrix_list[j].append(label_matrix[j])
            judge_matrix_list[j].append(judge_matrix[j])
        label_list.append(label)
    # 合并
    image_batch = torch.stack(image_list, dim=0)
    for i in range(len(label_matrix_list)):
        label_matrix_list[i] = torch.stack(label_matrix_list[i], dim=0)
        judge_matrix_list[i] = torch.stack(judge_matrix_list[i], dim=0)
    return image_batch, label_matrix_list, judge_matrix_list, label_list
