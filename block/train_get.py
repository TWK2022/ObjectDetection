import cv2
import tqdm
import wandb
import torch
import numpy as np
from block.val_get import val_get
from block.ModelEMA import ModelEMA
from block.lr_get import adam, lr_adjust
import tqdm
import torch


def train_get(args, data_dict, model_dict, loss):
    # 加载模型
    model = model_dict['model'].to(args.device, non_blocking=args.latch)
    # 学习率
    optimizer = adam(args.regularization, args.r_value, model.parameters(), lr=args.lr_start, betas=(0.937, 0.999))
    optimizer.load_state_dict(model_dict['optimizer_state_dict']) if model_dict['optimizer_state_dict'] else None
    optimizer_adjust = lr_adjust(args, model_dict['lr_adjust_item'])  # 学习率调整函数
    optimizer = optimizer_adjust(optimizer, model_dict['epoch'] + 1, 0)  # 初始化学习率
    # 使用平均指数移动(EMA)调整参数(不能将ema放到args中，否则会导致模型保存出错)
    ema = ModelEMA(model) if args.ema else None
    if args.ema:
        ema.updates = model_dict['ema_updates']
    # 数据集
    train_dataset = torch_dataset(args, 'train', data_dict['train'])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    train_shuffle = False if args.distributed else True  # 分布式设置sampler后shuffle要为False
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=train_shuffle,
                                                   drop_last=True, pin_memory=args.latch, num_workers=args.num_worker,
                                                   sampler=train_sampler, collate_fn=train_dataset.collate_fn)
    val_dataset = torch_dataset(args, 'val', data_dict['val'])
    val_sampler = None  # 分布式时数据合在主GPU上进行验证
    val_batch = args.batch // args.gpu_number  # 分布式验证时batch要减少为一个GPU的量
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch, shuffle=False, drop_last=False,
                                                 pin_memory=args.latch, num_workers=args.num_worker,
                                                 sampler=val_sampler, collate_fn=val_dataset.collate_fn)
    # 分布式初始化
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank) if args.distributed else model
    # wandb
    if args.wandb and args.local_rank == 0:
        wandb_image_list = []  # 记录所有的wandb_image最后一起添加(最多添加args.wandb_image_num张)
        wandb_class_name = {}  # 用于给边框添加标签名字
        for i in range(len(data_dict['class'])):
            wandb_class_name[i] = data_dict['class'][i]
    epoch_base = model_dict['epoch'] + 1  # 新的一轮要+1
    for epoch in range(epoch_base, epoch_base + args.epoch):
        # 训练
        print(f'\n-----------------------第{epoch}轮-----------------------') if args.local_rank == 0 else None
        model.train()
        train_loss = 0  # 记录训练损失
        train_frame_loss = 0  # 记录边框损失
        train_confidence_loss = 0  # 记录置信度框损失
        train_class_loss = 0  # 记录类别损失
        tqdm_show = tqdm.tqdm(total=len(data_dict['train']) // args.batch // args.gpu_number * args.gpu_number,
                              postfix=dict, mininterval=0.2) if args.local_rank == 0 else None  # tqdm
        for item, (image_batch, true_batch, judge_batch, label_list) in enumerate(train_dataloader):
            if args.wandb and args.local_rank == 0 and len(wandb_image_list) < args.wandb_image_num:
                wandb_image_batch = (image_batch * 255).cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)
            image_batch = image_batch.to(args.device, non_blocking=args.latch)  # 将输入数据放到设备上
            for i in range(len(true_batch)):  # 将标签矩阵放到对应设备上
                true_batch[i] = true_batch[i].to(args.device, non_blocking=args.latch)
            if args.amp:
                with torch.cuda.amp.autocast():
                    pred_batch = model(image_batch)
                    loss_batch, frame_loss, confidence_loss, class_loss = loss(pred_batch, true_batch, judge_batch)
                optimizer.zero_grad()
                args.amp.scale(loss_batch).backward()
                args.amp.step(optimizer)
                args.amp.update()
            else:
                pred_batch = model(image_batch)
                loss_batch, frame_loss, confidence_loss, class_loss = loss(pred_batch, true_batch, judge_batch)
                optimizer.zero_grad()
                loss_batch.backward()
                optimizer.step()
            # 调整参数，ema.updates会自动+1
            ema.update(model) if args.ema else None
            # 记录损失
            train_loss += loss_batch.item()
            train_frame_loss += frame_loss.item()
            train_confidence_loss += confidence_loss.item()
            train_class_loss += class_loss.item()
            # tqdm
            if args.local_rank == 0:
                tqdm_show.set_postfix({'当前loss': loss_batch.item()})  # 添加loss显示
                tqdm_show.update(args.gpu_number)  # 更新进度条
            # wandb
            if args.wandb and args.local_rank == 0 and epoch == 0 and len(wandb_image_list) < args.wandb_image_num:
                for i in range(len(wandb_image_batch)):  # 遍历每一张图片
                    image = wandb_image_batch[i]
                    frame = label_list[i][:, 0:4] / args.input_size  # (Cx,Cy,w,h)相对坐标
                    frame[:, 0:2] = frame[:, 0:2] - frame[:, 2:4] / 2
                    frame[:, 2:4] = frame[:, 0:2] + frame[:, 2:4]  # (x_min,y_min,x_max,y_max)相对坐标
                    cls = torch.argmax(label_list[i][:, 5:], dim=1)
                    box_data = []
                    for i in range(len(frame)):
                        class_id = cls[i].item()
                        box_data.append({"position": {"minX": frame[i][0].item(),
                                                      "minY": frame[i][1].item(),
                                                      "maxX": frame[i][2].item(),
                                                      "maxY": frame[i][3].item()},
                                         "class_id": class_id,
                                         "box_caption": wandb_class_name[class_id]})
                    wandb_image = wandb.Image(image, boxes={"predictions": {"box_data": box_data,
                                                                            'class_labels': wandb_class_name}})
                    wandb_image_list.append(wandb_image)
                    if len(wandb_image_list) == args.wandb_image_num:
                        break
        # tqdm
        tqdm_show.close() if args.local_rank == 0 else None
        # 计算平均损失
        train_loss = train_loss / (item + 1)
        train_frame_loss = train_frame_loss / (item + 1)
        train_confidence_loss = train_confidence_loss / (item + 1)
        train_class_loss = train_class_loss / (item + 1)
        print('\n| 轮次:{} | train_loss:{:.4f} | train_frame_loss:{:.4f} | train_confidence_loss:{:.4f} |'
              ' train_class_loss:{:.4f} | lr:{:.6f} |\n'
              .format(epoch + 1, train_loss, train_frame_loss, train_confidence_loss, train_class_loss,
                      optimizer.param_groups[0]['lr']))
        # 调整学习率
        optimizer = optimizer_adjust(optimizer, epoch + 1, train_loss)
        # 清理显存空间
        del image_batch, true_batch, judge_batch, pred_batch, loss_batch
        torch.cuda.empty_cache()
        # 验证
        if args.local_rank == 0:  # 分布式时只验证一次
            val_loss, val_frame_loss, val_confidence_loss, val_class_loss, precision, recall, m_ap \
                = val_get(args, val_dataloader, model, loss, ema)
        # 保存
        if args.local_rank == 0:  # 分布式时只保存一次
            model_dict['model'] = model.module if args.distributed else model.eval()
            model_dict['epoch'] = epoch
            model_dict['optimizer_state_dict'] = optimizer.state_dict()
            model_dict['lr_adjust_item'] = optimizer_adjust.lr_adjust_item
            model_dict['ema_updates'] = ema.updates if args.ema else model_dict['ema_updates']
            model_dict['class'] = data_dict['class']
            model_dict['train_loss'] = train_loss
            model_dict['val_loss'] = val_loss
            model_dict['val_m_ap'] = m_ap
            torch.save(model_dict, 'last.pt' if not args.prune else 'prune_last.pt')  # 保存最后一次训练的模型
            if m_ap > 0.1 and m_ap > model_dict['standard']:
                model_dict['standard'] = m_ap
                save_path = args.save_path if not args.prune else args.prune_save
                torch.save(model_dict, save_path)  # 保存最佳模型
                print('\n| 保存最佳模型:{} | val_m_ap:{:.4f} |\n'.format(args.save_path, m_ap))
            # wandb
            if args.wandb:
                wandb_log = {}
                if epoch == 0:
                    wandb_log.update({f'image/train_image': wandb_image_list})
                wandb_log.update({'train/train_loss': train_loss,
                                  'train/train_frame_loss': train_frame_loss,
                                  'train/train_confidence_loss': train_confidence_loss,
                                  'train/train_class_loss': train_class_loss,
                                  'val_loss/val_loss': val_loss,
                                  'val_loss/val_frame_loss': val_frame_loss,
                                  'val_loss/val_confidence_loss': val_confidence_loss,
                                  'val_loss/val_class_loss': val_class_loss,
                                  'val_metric/val_precision': precision,
                                  'val_metric/val_recall': recall,
                                  'val_metric/val_m_ap': m_ap})
                args.wandb_run.log(wandb_log)
        torch.distributed.barrier() if args.distributed else None  # 分布式时每轮训练后让所有GPU进行同步，快的GPU会在此等待
    return model_dict


class torch_dataset(torch.utils.data.Dataset):
    def __init__(self, args, tag, data):
        self.output_num = (3, 3, 3)  # 输出层数量，如(3, 3, 3)代表有三个大层，每层有三个小层
        self.stride = (8, 16, 32)  # 每个输出层尺寸缩小的幅度
        self.wh_multiple = 4  # 宽高的倍数，真实wh=网络原始输出[0-1]*倍数*anchor
        self.input_size = args.input_size  # 输入尺寸，如640
        self.output_class = args.output_class  # 输出类别数
        self.label_smooth = args.label_smooth  # 标签平滑，如(0.05,0.95)
        self.output_size = [int(self.input_size // i) for i in self.stride]  # 每个输出层的尺寸，如(80,40,20)
        self.anchor = (((12, 16), (19, 36), (40, 28)), ((36, 75), (76, 55), (72, 146)),
                       ((142, 110), (192, 243), (459, 401)))
        self.tag = tag  # 用于区分是训练集还是验证集
        self.data = data
        self.mosaic = args.mosaic
        self.mosaic_flip = args.mosaic_flip
        self.mosaic_screen = args.mosaic_screen

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 图片和标签处理，边框坐标处理为真实的Cx,Cy,w,h(归一化、减均值、除以方差、调维度等在模型中完成)
        if self.tag == 'train' and torch.rand(1) < self.mosaic:
            index_mix = torch.randperm(len(self.data))[0:4]
            index_mix[0] = index
            image, frame, cls_all = self._mosaic(index_mix, self.mosaic_screen)  # 马赛克增强、缩放和填充图片，相对坐标变为真实坐标(Cx,Cy,w,h)
        else:
            image = cv2.imdecode(np.fromfile(self.data[index][0], dtype=np.uint8), cv2.IMREAD_COLOR)  # 读取图片(可以读取中文)
            label = self.data[index][1].copy()  # 读取原始标签([:,类别号+Cx,Cy,w,h]，边框为相对边长的比例值)
            image, frame = self._resize(image.astype(np.uint8), label[:, 1:])  # 缩放和填充图片，相对坐标(Cx,Cy,w,h)变为真实坐标
            cls_all = label[:, 0]  # 类别号
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)  # 转为RGB通道
        image = (torch.tensor(image, dtype=torch.float32) / 255).permute(2, 0, 1)
        # 边框:转换为张量
        frame = torch.tensor(frame, dtype=torch.float32)
        # 置信度:为1
        confidence = torch.ones((len(frame), 1), dtype=torch.float32)
        # 类别:类别独热编码
        cls = torch.full((len(cls_all), self.output_class), self.label_smooth[0], dtype=torch.float32)
        for i in range(len(cls_all)):
            cls[i][int(cls_all[i])] = self.label_smooth[1]
        # 合并为标签
        label = torch.concat([frame, confidence, cls], dim=1).type(torch.float32)  # (Cx,Cy,w,h)真实坐标
        # 标签矩阵处理
        label_matrix_list = [0 for _ in range(len(self.output_num))]  # 存放每个输出层的标签矩阵，(Cx,Cy,w,h)真实坐标
        judge_matrix_list = [0 for _ in range(len(self.output_num))]  # 存放每个输出层的判断矩阵
        for i in range(len(self.output_num)):  # 遍历每个输出层
            label_matrix = torch.zeros(self.output_num[i], self.output_size[i], self.output_size[i],
                                       5 + self.output_class, dtype=torch.float32)  # 标签矩阵
            judge_matrix = torch.zeros(self.output_num[i], self.output_size[i], self.output_size[i],
                                       dtype=torch.bool)  # 判断矩阵，False代表没有标签
            if len(label) > 0:  # 存在标签
                frame = label[:, 0:4].clone()
                frame[:, 0:2] = frame[:, 0:2] / self.stride[i]
                frame[:, 2:4] = frame[:, 2:4] / self.wh_multiple
                # 标签对应输出网格的坐标
                Cx = frame[:, 0]
                x_grid = Cx.type(torch.int8)
                x_move = Cx - x_grid
                x_grid_add = x_grid + 2 * torch.round(x_move).type(torch.int8) - 1  # 每个标签可以由相邻网格预测
                x_grid_add = torch.clamp(x_grid_add, 0, self.output_size[i] - 1)  # 网格不能超出范围(与x_grid重复的网格之后不会加入)
                Cy = frame[:, 1]
                y_grid = Cy.type(torch.int8)
                y_move = Cy - y_grid
                y_grid_add = y_grid + 2 * torch.round(y_move).type(torch.int8) - 1  # 每个标签可以由相邻网格预测
                y_grid_add = torch.clamp(y_grid_add, 0, self.output_size[i] - 1)  # 网格不能超出范围(与y_grid重复的网格之后不会加入)
                # 遍历每个输出层的小层
                for j in range(self.output_num[i]):
                    # 根据wh制定筛选条件
                    frame_change = frame.clone()
                    w = frame_change[:, 2] / self.anchor[i][j][0]  # 该值要在0-1该层才能预测(但0-0.0625太小可以舍弃)
                    h = frame_change[:, 3] / self.anchor[i][j][1]  # 该值要在0-1该层才能预测(但0-0.0625太小可以舍弃)
                    wh_screen = torch.where((0.0625 < w) & (w < 1) & (0.0625 < h) & (h < 1), True, False)  # 筛选可以预测的标签
                    # 将标签填入对应的标签矩阵位置
                    for k in range(len(label)):
                        if wh_screen[k]:  # 根据wh筛选
                            label_matrix[j, x_grid[k], y_grid[k]] = label[k]
                            judge_matrix[j, x_grid[k], y_grid[k]] = True
                    # 将扩充的标签填入对应的标签矩阵位置
                    for k in range(len(label)):
                        if wh_screen[k] and not judge_matrix[j, x_grid_add[k], y_grid[k]]:  # 需要该位置有空位
                            label_matrix[j, x_grid_add[k], y_grid[k]] = label[k]
                            judge_matrix[j, x_grid_add[k], y_grid[k]] = True
                        if wh_screen[k] and not judge_matrix[j, x_grid[k], y_grid_add[k]]:  # 需要该位置有空位
                            label_matrix[j, x_grid[k], y_grid_add[k]] = label[k]
                            judge_matrix[j, x_grid[k], y_grid_add[k]] = True
            # 存放每个输出层的结果
            label_matrix_list[i] = label_matrix
            judge_matrix_list[i] = judge_matrix
        return image, label_matrix_list, judge_matrix_list, label  # 真实坐标(Cx,Cy,w,h)

    def _mosaic(self, index_mix, screen=10):  # 马赛克增强，合并后w,h不能小于screen
        x_center = int((torch.rand(1) * 0.4 + 0.3) * self.input_size)  # 0.3-0.7。四张图片合并的中心点
        y_center = int((torch.rand(1) * 0.4 + 0.3) * self.input_size)  # 0.3-0.7。四张图片合并的中心点
        image_merge = np.full((self.input_size, self.input_size, 3), 127)  # 合并后的图片
        frame_all = []  # 记录边框真实坐标(Cx,Cy,w,h)
        cls_all = []  # 记录类别号
        for item, index in enumerate(index_mix):
            image = cv2.imdecode(np.fromfile(self.data[index][0], dtype=np.uint8), cv2.IMREAD_COLOR)  # 读取图片(可以读取中文)
            label = self.data[index][1].copy()  # 相对坐标(类别号,Cx,Cy,w,h)
            # 垂直翻转
            if torch.rand(1) < self.mosaic_flip:
                image = cv2.flip(image, 1)  # 垂直翻转图片
                label[:, 1] = 1 - label[:, 1]  # 坐标变换:Cx=w-Cx
            # 根据input_size缩放图片
            h, w, _ = image.shape
            scale = self.input_size / w
            w = w * scale
            h = h * scale
            # 再随机缩放图片
            scale_w = torch.rand(1) + 0.5  # 0.5-1.5
            scale_h = 1 + torch.rand(1) * 0.5 if scale_w > 1 else 1 - torch.rand(1) * 0.5  # h与w同时放大和缩小
            w = int(w * scale_w)
            h = int(h * scale_h)
            image = cv2.resize(image, (w, h))
            # 合并图片，坐标变为合并后的真实坐标(Cx,Cy,w,h)
            if item == 0:  # 左上
                x_add, y_add = min(x_center, w), min(y_center, h)
                image_merge[y_center - y_add:y_center, x_center - x_add:x_center] = image[h - y_add:h, w - x_add:w]
                label[:, 1] = label[:, 1] * w + x_center - w  # Cx
                label[:, 2] = label[:, 2] * h + y_center - h  # Cy
                label[:, 3:5] = label[:, 3:5] * (w, h)  # w,h
            elif item == 1:  # 右上
                x_add, y_add = min(self.input_size - x_center, w), min(y_center, h)
                image_merge[y_center - y_add:y_center, x_center:x_center + x_add] = image[h - y_add:h, 0:x_add]
                label[:, 1] = label[:, 1] * w + x_center  # Cx
                label[:, 2] = label[:, 2] * h + y_center - h  # Cy
                label[:, 3:5] = label[:, 3:5] * (w, h)  # w,h
            elif item == 2:  # 右下
                x_add, y_add = min(self.input_size - x_center, w), min(self.input_size - y_center, h)
                image_merge[y_center:y_center + y_add, x_center:x_center + x_add] = image[0:y_add, 0:x_add]
                label[:, 1] = label[:, 1] * w + x_center  # Cx
                label[:, 2] = label[:, 2] * h + y_center  # Cy
                label[:, 3:5] = label[:, 3:5] * (w, h)  # w,h
            else:  # 左下
                x_add, y_add = min(x_center, w), min(self.input_size - y_center, h)
                image_merge[y_center:y_center + y_add, x_center - x_add:x_center] = image[0:y_add, w - x_add:w]
                label[:, 1] = label[:, 1] * w + x_center - w  # Cx
                label[:, 2] = label[:, 2] * h + y_center  # Cy
                label[:, 3:5] = label[:, 3:5] * (w, h)  # w,h
            frame_all.append(label[:, 1:5])
            cls_all.append(label[:, 0])
        # 合并标签
        frame_all = np.concatenate(frame_all, axis=0)
        cls_all = np.concatenate(cls_all, axis=0)
        # 筛选掉不在图片内的标签
        frame_all[:, 0:2] = frame_all[:, 0:2] - frame_all[:, 2:4] / 2
        frame_all[:, 2:4] = frame_all[:, 0:2] + frame_all[:, 2:4]  # 真实坐标(x_min,y_min,x_max,y_max)
        frame_all = np.clip(frame_all, 0, self.input_size - 1)  # 压缩坐标到图片内
        frame_all[:, 2:4] = frame_all[:, 2:4] - frame_all[:, 0:2]
        frame_all[:, 0:2] = frame_all[:, 0:2] + frame_all[:, 2:4] / 2  # 真实坐标(Cx,Cy,w,h)
        judge_list = np.where((frame_all[:, 2] > screen) & (frame_all[:, 3] > screen), True, False)  # w,h不能小于screen
        frame_all = frame_all[judge_list]
        cls_all = cls_all[judge_list]
        return image_merge, frame_all, cls_all

    def _resize(self, image, frame):  # 将图片四周填充变为正方形，frame输入输出都为[[Cx,Cy,w,h]...](相对原图片的比例值)
        shape = image.shape
        w0 = shape[1]
        h0 = shape[0]
        if w0 == h0 == self.input_size:  # 不需要变形
            frame *= self.input_size
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

    def _draw(self, image, frame_all):  # 测试时画图使用，真实坐标(Cx,Cy,w,h)
        frame_all[:, 0:2] = frame_all[:, 0:2] - frame_all[:, 2:4] / 2
        frame_all[:, 2:4] = frame_all[:, 0:2] + frame_all[:, 2:4]  # 真实坐标(x_min,y_min,x_max,y_max)
        for frame in frame_all:
            x1, y1, x2, y2 = frame
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255, 0), thickness=2)
        cv2.imwrite('save_check.jpg', image)

    def collate_fn(self, getitem_batch):  # 自定义__getitem__合并方式
        image_list = []
        label_matrix_list = [[] for _ in range(len(getitem_batch[0][1]))]
        judge_matrix_list = [[] for _ in range(len(getitem_batch[0][2]))]
        label_list = []
        for i in range(len(getitem_batch)):  # 遍历所有__getitem__
            image = getitem_batch[i][0]
            label_matrix = getitem_batch[i][1]
            judge_matrix = getitem_batch[i][2]
            label = getitem_batch[i][3]
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
        return image_batch, label_matrix_list, judge_matrix_list, label_list  # 均为(Cx,Cy,w,h)真实坐标
