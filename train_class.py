import os
import cv2
import math
import copy
import wandb
import torch
import logging
import torchvision
import numpy as np
from model.layer import decode


class train_class:
    '''
        model_load: 加载模型
        data_load: 加载数据
        dataloader_load: 加载数据处理器
        optimizer_load: 加载学习率
        loss_load: 训练损失
        train: 训练模型
        validation: 训练时的模型验证
    '''

    def __init__(self, args):
        self.args = args
        self.model_dict = self.model_load()  # 模型
        self.model_dict['model'] = self.model_dict['model'].to(args.device, non_blocking=args.latch)  # 设备
        self.data_dict = self.data_load()  # 数据
        self.train_dataloader, self.val_dataloader, self.train_dataset = self.dataloader_load()  # 数据处理器
        self.optimizer, self.optimizer_adjust = self.optimizer_load()  # 学习率、学习率调整
        self.loss = self.loss_load()  # 损失函数
        if args.local_rank == 0 and args.ema:  # 平均指数移动(EMA)，创建ema模型
            self.ema = model_ema(self.model_dict['model'])
            self.ema.update_total = self.model_dict['ema_update']
        if args.distributed:  # 分布式初始化
            self.model_dict['model'] = torch.nn.parallel.DistributedDataParallel(self.model_dict['model'],
                                                                                 device_ids=[args.local_rank],
                                                                                 output_device=args.local_rank)
        if args.local_rank == 0 and args.log:  # 日志
            log_path = os.path.dirname(__file__) + '/log.log'
            logging.basicConfig(filename=log_path, level=logging.INFO,
                                format='%(asctime)s | %(levelname)s | %(message)s')
            logging.info('-------------------- log --------------------')
        if args.local_rank == 0 and args.wandb:  # wandb
            self.wandb_image_number = 16  # 记录图片数量
            self.wandb_image_list = []  # 记录所有的wandb_image最后一起添加
            self.wandb_class_name = {}  # 用于给边框添加标签名字
            for index, name in enumerate(self.data_dict['class']):
                self.wandb_class_name[index] = name

    @staticmethod
    def weight_assignment(model, prune_model):  # 剪枝模型权重赋值
        for module, prune_module in zip(model.modules(), prune_model.modules()):
            if not hasattr(module, 'weight'):  # 对权重层赋值
                continue
            weight = module.weight.data
            prune_weight = prune_module.weight.data
            if len(weight.shape) == 1:  # 单维权重(如bn层)
                prune_module.weight.data = weight[:prune_weight.shape[0]]
            else:  # 两维权重(如conv层)
                prune_module.weight.data = weight[:prune_weight.shape[0], :prune_weight.shape[1]]
        return prune_model

    @staticmethod
    def iou(pred, label):  # 输入(batch,(x_min,y_min,w,h))相对/真实坐标
        x1 = torch.maximum(pred[:, 0], label[:, 0])
        y1 = torch.maximum(pred[:, 1], label[:, 1])
        x2 = torch.minimum(pred[:, 0] + pred[:, 2], label[:, 0] + label[:, 2])
        y2 = torch.minimum(pred[:, 1] + pred[:, 3], label[:, 1] + label[:, 3])
        intersection = torch.clamp(x2 - x1, 0) * torch.clamp(y2 - y1, 0)
        union = pred[:, 2] * pred[:, 3] + label[:, 2] * label[:, 3] - intersection
        return intersection / union

    @staticmethod
    def nms(pred, iou_threshold):  # 输入(batch,(x_min,y_min,w,h))真实坐标
        pred[:, 2:4] = pred[:, 0:2] + pred[:, 2:4]  # (x_min,y_min,x_max,y_max)
        index = torchvision.ops.nms(pred[:, 0:4], pred[:, 4], 1 - iou_threshold)
        pred = pred[index]
        pred[:, 2:4] = pred[:, 2:4] - pred[:, 0:2]  # (x_min,y_min,w,h)
        return pred

    def model_load(self):
        args = self.args
        if os.path.exists(args.weight_path):
            model_dict = torch.load(args.weight_path, map_location='cpu', weights_only=False)
            for param in model_dict['model'].parameters():
                param.requires_grad_(True)  # 打开梯度(保存的ema模型为关闭)
            if args.weight_again:
                model_dict['epoch_finished'] = 0  # 已训练的轮数
                model_dict['optimizer_state_dict'] = None  # 学习率参数
                model_dict['ema_update'] = 0  # ema参数
                model_dict['standard'] = 0  # 评价指标
        else:  # 创建新模型
            if os.path.exists(args.prune_weight_path):
                model_dict = torch.load(args.prune_weight_path, map_location='cpu', weights_only=False)
                model = model_dict['model']  # 原模型
                exec(f'from model.{args.model} import {args.model}')
                config = self._bn_prune(model)  # 剪枝参数
                prune_model = eval(f'{args.model}(self.args, config=config)')  # 剪枝模型
                model = self.weight_assignment(model, prune_model)  # 剪枝模型赋值
            else:
                exec(f'from model.{args.model} import {args.model}')
                model = eval(f'{args.model}(self.args)')
            model_dict = {
                'model': model,
                'epoch_finished': 0,  # 已训练的轮数
                'optimizer_state_dict': None,  # 学习率参数
                'ema_update': 0,  # ema参数
                'standard': 0,  # 评价指标
            }
        return model_dict

    def data_load(self):
        args = self.args
        # 训练集[[图片路径,标签路径]...]
        with open(f'{args.data_path}/train.txt', encoding='utf-8') as f:
            train_list = [[f'{args.data_path}/{__}' for __ in _.strip().split(',')] for _ in f.readlines()]
        # 验证集[[图片路径,标签路径]...]
        with open(f'{args.data_path}/val.txt', encoding='utf-8') as f:
            val_list = [[f'{args.data_path}/{__}' for __ in _.strip().split(',')] for _ in f.readlines()]
        data_dict = {'train': train_list, 'val': val_list}
        return data_dict

    def dataloader_load(self):
        args = self.args
        # 数据集
        train_dataset = torch_dataset(args, 'train', self.data_dict['train'])
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
        train_shuffle = False if args.distributed else True  # 分布式设置sampler后shuffle要为False
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=train_shuffle,
                                                       drop_last=True, pin_memory=args.latch,
                                                       num_workers=args.num_worker,
                                                       sampler=train_sampler, collate_fn=train_dataset.collate_fn)
        val_dataset = torch_dataset(args, 'val', self.data_dict['val'])
        val_sampler = None  # 分布式时数据合在主GPU上进行验证
        val_batch = args.batch // args.device_number  # 分布式验证时batch要减少为一个GPU的量
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch, shuffle=False,
                                                     drop_last=False, pin_memory=args.latch,
                                                     num_workers=args.num_worker,
                                                     sampler=val_sampler, collate_fn=train_dataset.collate_fn)
        return train_dataloader, val_dataloader, train_dataset

    def optimizer_load(self):
        args = self.args
        if args.regularization == 'L2':
            optimizer = torch.optim.Adam(self.model_dict['model'].parameters(),
                                         lr=args.lr_start, betas=(0.937, 0.999), weight_decay=args.r_value)
        else:
            optimizer = torch.optim.Adam(self.model_dict['model'].parameters(),
                                         lr=args.lr_start, betas=(0.937, 0.999))
        if self.model_dict['optimizer_state_dict'] is not None:
            optimizer.load_state_dict(self.model_dict['optimizer_state_dict'])
        step_epoch = len(self.data_dict['train']) // args.batch // args.device_number * args.device_number  # 每轮步数
        optimizer_adjust = lr_adjust(args, step_epoch, self.model_dict['epoch_finished'])  # 学习率调整函数
        optimizer = optimizer_adjust(optimizer)  # 学习率初始化
        return optimizer, optimizer_adjust

    def loss_load(self):
        loss = loss_class(self.args)
        return loss

    def train(self):
        args = self.args
        model = self.model_dict['model']
        epoch_base = self.model_dict['epoch_finished'] + 1  # 新的一轮要+1
        for epoch in range(epoch_base, args.epoch + 1):
            if args.local_rank == 0:
                info = f'-----------------------epoch:{epoch}-----------------------'
                print(info) if args.print_info else None
            model.train()
            train_loss = 0  # 记录损失
            frame_loss = 0
            confidence_loss = 0
            class_loss = 0
            self.train_dataset.epoch_update(epoch)
            for index, (image_batch, screen_list, label_expend, label_list) in enumerate(self.train_dataloader):
                if args.local_rank == 0 and args.wandb and len(self.wandb_image_list) < self.wandb_image_number:
                    wandb_image_batch = (image_batch * 255).cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)
                image_batch = image_batch.to(args.device, non_blocking=args.latch)
                screen_list = [_.to(args.device, non_blocking=args.latch) for _ in screen_list]
                label_expend = label_expend.to(args.device, non_blocking=args.latch)
                if args.amp:
                    with torch.cuda.amp.autocast():
                        pred_batch = model(image_batch)
                        loss_batch, frame_, confidence_, class_ = self.loss(pred_batch, screen_list, label_expend)
                    args.amp.scale(loss_batch).backward()
                    args.amp.step(self.optimizer)
                    args.amp.update()
                    self.optimizer.zero_grad()
                else:
                    pred_batch = model(image_batch)
                    loss_batch, frame_, confidence_, class_ = self.loss(pred_batch, screen_list, label_expend)
                    loss_batch.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                self.ema.update(model) if args.local_rank == 0 and args.ema else None  # 更新ema模型参数
                train_loss += loss_batch.item()  # 记录损失
                frame_loss += frame_.item()
                confidence_loss += confidence_.item()
                class_loss += class_.item()
                self.optimizer = self.optimizer_adjust(self.optimizer)  # 调整学习率
                # wandb
                if args.local_rank == 0 and args.wandb and len(self.wandb_image_list) < self.wandb_image_number:
                    for image, label in zip(wandb_image_batch, label_list):  # 遍历每一张图片
                        frame = label[:, 0:4] / args.input_size  # (cx,cy,w,h)相对坐标
                        frame[:, 0:2] = frame[:, 0:2] - frame[:, 2:4] / 2
                        frame[:, 2:4] = frame[:, 0:2] + frame[:, 2:4]  # (x_min,y_min,x_max,y_max)相对坐标
                        cls = torch.argmax(label[:, 5:], dim=1)
                        box_data = []
                        for index_, frame_ in enumerate(frame):
                            class_id = cls[index_].item()
                            box_data.append({"position": {"minX": frame_[0].item(),
                                                          "minY": frame_[1].item(),
                                                          "maxX": frame_[2].item(),
                                                          "maxY": frame_[3].item()},
                                             "class_id": class_id,
                                             "box_caption": self.wandb_class_name[class_id]})
                        wandb_image = wandb.Image(image, boxes={"predictions": {"box_data": box_data,
                                                                                'class_labels': self.wandb_class_name}})
                        self.wandb_image_list.append(wandb_image)
                    self.wandb_image_list = self.wandb_image_list[:self.wandb_image_number]
            # 计算平均损失
            train_loss /= index + 1
            frame_loss /= index + 1
            confidence_loss /= index + 1
            class_loss /= index + 1
            # 日志
            if args.local_rank == 0 and args.print_info:
                info = (f'| train | train_loss:{train_loss:.4f} | frame_loss:{frame_loss:.4f} |'
                        f' confidence_loss:{confidence_loss:.4f} | class_loss:{class_loss:.4f} |'
                        f' lr:{self.optimizer.param_groups[0]["lr"]:.6f} |')
                print(info)
            # 清理显存空间
            del image_batch, screen_list, label_expend, label_list, pred_batch, loss_batch
            torch.cuda.empty_cache()
            # 验证
            if args.local_rank == 0:  # 分布式时只验证一次
                val_loss, precision, recall, m_ap = self.validation()
            # 保存
            if args.local_rank == 0:  # 分布式时只保存一次
                self.model_dict['model'] = self.ema.ema_model if args.ema else (
                    model.module if args.distributed else model)
                self.model_dict['epoch_finished'] = epoch
                self.model_dict['optimizer_state_dict'] = self.optimizer.state_dict()
                self.model_dict['ema_update'] = self.ema.update_total if args.ema else self.model_dict['ema_update']
                self.model_dict['class'] = self.data_dict['class']
                self.model_dict['train_loss'] = train_loss
                self.model_dict['val_loss'] = val_loss
                self.model_dict['val_precision'] = precision
                self.model_dict['val_recall'] = recall
                self.model_dict['val_m_ap'] = m_ap
                if epoch % args.save_epoch == 0 or epoch == args.epoch:
                    torch.save(self.model_dict, args.save_path)  # 保存模型
                if m_ap >= self.model_dict['standard'] and m_ap >= 0.25:
                    self.model_dict['standard'] = m_ap
                    torch.save(self.model_dict, args.save_best)  # 保存最佳模型
                    if args.local_rank == 0:  # 日志
                        info = (f'| best_model | val_loss:{val_loss:.4f} |'
                                f' threshold:{args.confidence_threshold:.2f}+{args.iou_threshold:.2f} |'
                                f' val_precision:{precision:.4f} | val_recall:{recall:.4f} | val_m_ap:{m_ap:.4f} |')
                        print(info) if args.print_info else None
                        logging.info(info) if args.log else None
                # wandb
                if args.wandb:
                    wandb_log = {}
                    if epoch == 0:
                        wandb_log.update({f'image/train_image': self.wandb_image_list})
                    wandb_log.update({'metric/train_loss': train_loss,
                                      'metric/val_loss': val_loss,
                                      'metric/val_m_ap': m_ap,
                                      'metric/val_precision': precision,
                                      'metric/val_recall': recall})
                    args.wandb_run.log(wandb_log)
            torch.distributed.barrier() if args.distributed else None  # 分布式时每轮训练后让所有GPU进行同步，快的GPU会在此等待

    def validation(self):
        args = self.args
        with torch.no_grad():
            model = self.ema.ema_model.eval() if args.ema else self.model_dict['model'].eval()
            val_loss = 0  # 记录验证损失
            tp_all = 0
            fp_all = 0
            fn_all = 0
            for index, (image_batch, screen_list, label_expend, label_list) in enumerate(self.val_dataloader):
                image_batch = image_batch.to(args.device, non_blocking=args.latch)
                screen_list = [_.to(args.device, non_blocking=args.latch) for _ in screen_list]
                label_expend = label_expend.to(args.device, non_blocking=args.latch)
                pred_batch = model(image_batch)
                loss_batch, frame_loss, confidence_loss, class_loss = self.loss(pred_batch.clone(), screen_list,
                                                                                label_expend)
                val_loss += loss_batch.item()
                # 计算批量指标
                tp, fn, fp = self.metric(pred_batch, label_list, args.confidence_threshold, args.iou_threshold)
                tp_all += tp
                fn_all += fn
                fp_all += fp
            # 计算指标
            val_loss /= index + 1
            precision = tp_all / (tp_all + fp_all + 1e-6)
            recall = tp_all / (tp_all + fn_all + 1e-6)
            m_ap = precision * recall
            # 日志
            info = (f'| val | val_loss:{val_loss:.4f} |'
                    f' threshold:{args.confidence_threshold:.2f}+{args.iou_threshold:.2f} |'
                    f' val_precision:{precision:.4f} | val_recall:{recall:.4f} | val_m_ap:{m_ap:.4f} |')
            print(info) if args.print_info else None
        return val_loss, precision, recall, m_ap

    def metric(self, pred_batch, label_list, confidence_threshold, iou_threshold):  # (batch,(cx,cy,w,h,confidence...))
        tp_all = 0
        fn_all = 0
        fp_all = 0
        for pred, label in zip(pred_batch, label_list):  # 遍历每张图片
            pred[:, 0:2] = pred[:, 0:2] - pred[:, 2:4] / 2  # (x_min,y_min,w,h)
            label[:, 0:2] = label[:, 0:2] - label[:, 2:4] / 2  # (x_min,y_min,w,h)
            pred = pred[pred[:, 4] > confidence_threshold]  # 置信度筛选
            if len(pred) == 0:  # 没有预测值
                fn_all += len(label)
                continue
            pred = self.nms(pred, iou_threshold)[:100]  # 非极大值抑制，最多100
            if len(label) == 0:  # 图片没有标签
                fp_all += len(pred)
                continue
            # 计算指标
            pred_cls = torch.argmax(pred[:, 5:], dim=1)
            true_cls = torch.argmax(label[:, 5:], dim=1)
            tp = 0
            for index, target in enumerate(label):
                iou_all = self.iou(pred, target.unsqueeze(0))
                screen_tp = torch.where((iou_all > iou_threshold) & (pred_cls == true_cls[index]), True, False)
                tp += min(len(pred[screen_tp]), 1)  # 一个标签只有一个预测值
            fp = len(pred) - tp
            fn = len(label) - tp
            # 记录
            tp_all += tp
            fn_all += fn
            fp_all += fp
        return tp_all, fn_all, fp_all

    def _bn_prune(self, model):  # 通过bn层裁剪模型
        args = self.args
        weight = []  # 权重
        weight_layer = []  # 每个权重所在的层
        layer = 0  # 层数记录
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                weight.append(module.weight.data.clone())
                weight_layer.append(np.full((len(module.weight.data),), layer))
                layer += 1
        weight_abs = torch.concatenate(weight, dim=0).abs()
        weight_index = np.concatenate(weight_layer, axis=0)
        # 剪枝
        boundary = int(len(weight_abs) * args.prune_ratio)
        weight_index_keep = weight_index[np.argsort(weight_abs)[-boundary:]]  # 保留的参数所在的层数
        config = []  # 裁剪结果
        for layer, weight_one in enumerate(weight):
            layer_number = max(np.sum(weight_index_keep == layer).item(), 1)  # 剪枝后该层的参数个数，至少1个
            config.append(layer_number)
        return config

    def _check_draw(self, image, frame_all):  # 测试时画图使用，真实坐标(cx,cy,w,h)
        frame_all[:, 0:2] = frame_all[:, 0:2] - frame_all[:, 2:4] / 2
        frame_all[:, 2:4] = frame_all[:, 0:2] + frame_all[:, 2:4]  # 真实坐标(x_min,y_min,x_max,y_max)
        for frame in frame_all:
            x1, y1, x2, y2 = frame
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255, 0), thickness=2)
        cv2.imwrite('check_draw.jpg', image)


class loss_class():
    def __init__(self, args):
        self.device = args.device
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.label_smooth = args.label_smooth
        self.frame_loss = self.ciou  # 边框损失函数
        self.confidence_loss = focal_loss()  # 置信度损失函数
        if args.output_class == 1:  # 分类损失函数
            self.class_loss = torch.nn.BCEWithLogitsLoss()
        else:
            self.class_loss = torch.nn.CrossEntropyLoss()

    def __call__(self, pred, screen_list, label_expend):
        pred[:, :, 4:] = torch.log(pred[:, :, 4:] / (1 - pred[:, :, 4:]))  # 逆sigmoid
        pred_need = []  # 有标签对应的区域
        confidence_label = torch.full_like(pred[:, :, 4], self.label_smooth, device=pred.device)  # 置信度标签
        for index, (pred_, screen) in enumerate(zip(pred, screen_list)):
            pred_need.append(pred_[screen])
            one = confidence_label[index]
            one[screen] = 1 - self.label_smooth
            confidence_label[index] = one
        pred_need = torch.concat(pred_need)
        confidence_loss = self.confidence_loss(pred[:, :, 4], confidence_label)  # 置信度损失(计算所有的)
        if len(pred_need) == 0:  # 没有标签
            frame_loss = torch.tensor(1)
            class_loss = torch.tensor(1)
        else:
            frame_loss = 1 - torch.mean(self.frame_loss(pred_need[:, 0:4], label_expend[:, 0:4]))  # 边框损失(只计算需要的)
            class_loss = self.class_loss(pred_need[:, 5:], label_expend[:, 5:])  # 分类损失(只计算需要的)
        loss = frame_loss + confidence_loss + class_loss
        return loss, frame_loss, confidence_loss, class_loss

    @staticmethod
    def iou(pred, label):  # 输入(batch,(x_min,y_min,w,h))相对/真实坐标
        x1 = torch.maximum(pred[:, 0], label[:, 0])
        y1 = torch.maximum(pred[:, 1], label[:, 1])
        x2 = torch.minimum(pred[:, 0] + pred[:, 2], label[:, 0] + label[:, 2])
        y2 = torch.minimum(pred[:, 1] + pred[:, 3], label[:, 1] + label[:, 3])
        intersection = torch.clamp(x2 - x1, 0) * torch.clamp(y2 - y1, 0)
        union = pred[:, 2] * pred[:, 3] + label[:, 2] * label[:, 3] - intersection
        return intersection / union

    @staticmethod
    def L1_L2(pred, label):  # 输入(batch,(x_min,y_min,w,h))相对/真实坐标
        x1 = torch.minimum(pred[:, 0], label[:, 0])
        y1 = torch.minimum(pred[:, 1], label[:, 1])
        x2 = torch.maximum(pred[:, 0] + pred[:, 2], label[:, 0] + label[:, 2])
        y2 = torch.maximum(pred[:, 1] + pred[:, 3], label[:, 1] + label[:, 3])
        L1 = torch.square(pred[:, 0] - label[:, 0]) + torch.square(pred[:, 1] - label[:, 1])
        L2 = torch.square(x2 - x1) + torch.square(y2 - y1)
        return L1 / L2

    def ciou(self, pred, label):  # 输入(batch,(cx,cy,w,h))相对/真实坐标
        pred = pred.clone()
        label = label.clone()
        pred[:, 0:2] = pred[:, 0:2] - pred[:, 2:4] / 2  # (x_min,y_min,w,h)
        label[:, 0:2] = label[:, 0:2] - label[:, 2:4] / 2  # (x_min,y_min,w,h)
        iou = self.iou(pred, label)
        L1_L2 = self.L1_L2(pred, label)
        v = (4 / (3.1415926 ** 2)) * torch.pow(
            torch.atan(label[:, 2] / label[:, 3]) - torch.atan(pred[:, 2] / pred[:, 3]), 2)
        with torch.no_grad():
            alpha = v / (v + 1 - iou + 0.00001)
        return iou - L1_L2 - alpha * v


class focal_loss(torch.nn.Module):  # 聚焦损失
    def __init__(self, alpha=0.25, gamma=2, scale=200):
        super().__init__()
        self.alpha = alpha  # 增大更关注正样本
        self.gamma = gamma  # 增大更关注难样本
        self.scale = scale  # 整体提高置信度损失权重

    def forward(self, pred, label):
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, label, reduction='none')
        alpha = self.alpha * label + (1 - self.alpha) * (1 - label)
        loss = self.scale * torch.mean(alpha * (1 - torch.exp(-bce_loss)) ** self.gamma * bce_loss)
        return loss


class model_ema:
    def __init__(self, model, decay=0.9999, tau=2000, update_total=0):
        self.ema_model = copy.deepcopy(self._get_model(model)).eval()  # FP32 EMA
        self.update_total = update_total
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for param in self.ema_model.parameters():
            param.requires_grad_(False)  # 关闭梯度

    def update(self, model):
        with torch.no_grad():
            self.update_total += 1
            d = self.decay(self.update_total)
            state_dict = self._get_model(model).state_dict()
            for k, v in self.ema_model.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * state_dict[k].detach()

    def _get_model(self, model):
        if type(model) in (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel):
            return model.module
        else:
            return model


class lr_adjust:
    def __init__(self, args, step_epoch, epoch_finished):
        self.lr_start = args.lr_start  # 初始学习率
        self.lr_end = args.lr_end_ratio * args.lr_start  # 最终学习率
        self.lr_end_epoch = args.lr_end_epoch  # 最终学习率达到的轮数
        self.step_all = self.lr_end_epoch * step_epoch  # 总调整步数
        self.step_finished = epoch_finished * step_epoch  # 已调整步数
        self.warmup_step = max(5, int(args.warmup_ratio * self.step_all))  # 预热训练步数

    def __call__(self, optimizer):
        self.step_finished += 1
        step_now = self.step_finished
        decay = step_now / self.step_all
        lr = self.lr_end + (self.lr_start - self.lr_end) * math.cos(math.pi / 2 * decay)
        if step_now <= self.warmup_step:
            lr = lr * (0.1 + 0.9 * step_now / self.warmup_step)
        lr = max(lr, 0.000001)
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr
        return optimizer


class torch_dataset(torch.utils.data.Dataset):
    def __init__(self, args, tag, data):
        self.tag = tag
        self.data = data
        self.epoch_total = args.epoch
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.output_class = args.output_class
        self.output_layer = args.output_layer
        self.label_smooth = args.label_smooth
        self.mosaic = args.mosaic
        self.mosaic_probability = 0
        self.mosaic_flip = 0.5  # hsv通道随机变换概率
        self.mosaic_hsv = 0.5  # 垂直翻转概率
        self.mosaic_screen = 10  # 增强后框的w,h不能小于mosaic_screen
        # 每个位置的预测范围(模型原始输出通常在+-4之前，sigmoid后约为0.01-0.99)
        range_floor = [torch.full((1, self.output_layer[_], self.output_size[_], self.output_size[_], 5), -4,
                                  dtype=torch.float32) for _ in range(len(self.output_size))]  # 输出下限
        range_upper = [torch.full_like(_, 4) for _ in range_floor]  # 输出上限
        with torch.no_grad():
            model_decode = decode(args)
            self.range_floor = model_decode(range_floor)[0]  # 预测下限
            self.range_upper = model_decode(range_upper)[0]  # 预测上限

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 图片和标签处理，边框坐标处理为真实的cx,cy,w,h(归一化、减均值、除以方差、调维度等在模型中完成)
        if self.tag == 'train' and torch.rand(1) < self.mosaic_probability:
            index_mix = torch.randperm(len(self.data))[0:4]
            index_mix[0] = index
            image, frame, cls_all = self._mosaic(index_mix)  # 马赛克增强、缩放和填充图片，相对坐标变为真实坐标(cx,cy,w,h)
        else:
            image = cv2.imdecode(np.fromfile(self.data[index][0], dtype=np.uint8), cv2.IMREAD_COLOR)  # 读取图片
            with open(self.data[index][1], 'r', encoding='utf-8') as f:
                label = np.array([_.strip().split() for _ in f.readlines()], dtype=np.float32)  # 相对坐标(类别号,cx,cy,w,h)
            if len(label) > 0:  # 有标签
                image, frame = self._resize(image.astype(np.uint8), label[:, 1:])  # 缩放和填充图片，相对坐标(cx,cy,w,h)变为真实坐标
                cls_all = label[:, 0]  # 类别号
            else:
                image = np.resize(image.astype(np.uint8), (self.input_size, self.input_size, 3))
                frame = None
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)  # 转为RGB通道
        # self.draw_image(image, frame)  # 画图检查
        image = (torch.tensor(image, dtype=torch.float32) / 255).permute(2, 0, 1)
        if frame is None:
            label_screen = torch.tensor([], dtype=torch.int32)
            label_expend = torch.tensor([], dtype=torch.int32)
            return image, label_screen, label_expend, label
        # 原始标签
        label_number = len(frame)  # 标签数量
        frame = torch.tensor(frame, dtype=torch.float32)  # 边框
        confidence = torch.full((label_number, 1), 1 - self.label_smooth, dtype=torch.float32)  # 置信度
        cls = torch.full((label_number, self.output_class), self.label_smooth, dtype=torch.float32)  # 类别独热编码
        for index in range(label_number):
            cls[index][int(cls_all[index])] = 1 - self.label_smooth
        label = torch.concat([frame, confidence, cls], dim=1).type(torch.float32)  # (cx,cy,w,h)真实坐标
        # 标签处理
        label_expend = torch.zeros((len(self.range_floor), 5 + self.output_class), dtype=torch.float32)
        label_screen = []
        label_set = set()
        for label_ in label:
            cx, cy, w, h = label_[0:4]
            screen = ((self.range_floor[:, 0] < cx) & (cx < self.range_upper[:, 0])
                      & (self.range_floor[:, 1] < cy) & (cy < self.range_upper[:, 1])
                      & (self.range_floor[:, 2] < w) & (w < self.range_upper[:, 2])
                      & (self.range_floor[:, 3] < h) & (h < self.range_upper[:, 3]))
            index = torch.nonzero(screen)
            if len(index) > 0:
                index = set(index[:, 0].tolist())
            else:
                index = set()
            intersection = label_set.intersection(index)
            if intersection:  # 标签冲突，保留更近的标签
                index -= intersection
                intersection = list(intersection)
                for index_ in intersection:
                    old = label_expend[index_]
                    center = (self.range_floor[index_][0:2] + self.range_upper[index_][0:2]) / 2  # 预测中心值
                    if torch.linalg.norm(old[0:2] - center) > torch.linalg.norm(label_[0:2] - center):  # 新的标签更近
                        label_expend[index_] = label_
            index = list(index)
            label_expend[index] = label_
            label_screen.extend(index)
        label_screen, _ = torch.sort(torch.tensor(label_screen))  # 标签筛选
        label_expend = label_expend[label_screen]  # 标签
        return image, label_screen, label_expend, label

    def collate_fn(self, getitem_list):  # 自定义__getitem__合并方式
        image_list = []
        screen_list = []
        label_expend_list = []
        label_list = []
        for i in range(len(getitem_list)):  # 遍历所有__getitem__
            image = getitem_list[i][0]
            screen_matrix = getitem_list[i][1]
            label_expend = getitem_list[i][2]
            label = getitem_list[i][3]
            image_list.append(image)
            screen_list.append(screen_matrix)
            label_expend_list.append(label_expend)
            label_list.append(label)
        # 合并
        image_batch = torch.stack(image_list, dim=0)
        label_expend = torch.concat(label_expend_list, dim=0)
        return image_batch, screen_list, label_expend, label_list  # 均为(cx,cy,w,h)真实坐标

    def epoch_update(self, epoch_now):  # 根据轮数进行调整
        if 0.1 * self.epoch_total < epoch_now < 0.9 * self.epoch_total:  # 开始和末尾不加噪
            self.mosaic_probability = self.mosaic
        else:
            self.mosaic_probability = 0

    def _mosaic(self, index_mix):  # 马赛克增强，合并后w,h不能小于screen
        x_center = int((torch.rand(1) * 0.4 + 0.3) * self.input_size)  # 0.3-0.7。四张图片合并的中心点
        y_center = int((torch.rand(1) * 0.4 + 0.3) * self.input_size)  # 0.3-0.7。四张图片合并的中心点
        image_merge = np.full((self.input_size, self.input_size, 3), 128)  # 合并后的图片
        frame_all = []  # 记录边框真实坐标(cx,cy,w,h)
        cls_all = []  # 记录类别号
        for i, index in enumerate(index_mix):
            image = cv2.imdecode(np.fromfile(self.data[index][0], dtype=np.uint8), cv2.IMREAD_COLOR)  # 读取图片
            with open(self.data[index][1], 'r', encoding='utf-8') as f:
                label = np.array([_.strip().split() for _ in f.readlines()], dtype=np.float32)  # 相对坐标(类别号,cx,cy,w,h)
            # hsv通道变换
            if torch.rand(1) < self.mosaic_hsv:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
                image[:, :, 0] += np.random.rand(1) * 60 - 30  # -30到30
                image[:, :, 1] += np.random.rand(1) * 60 - 30  # -30到30
                image[:, :, 2] += np.random.rand(1) * 60 - 30  # -30到30
                image = np.clip(image, 0, 255).astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
            # 垂直翻转
            if torch.rand(1) < self.mosaic_flip:
                image = cv2.flip(image, 1)  # 垂直翻转图片
                label[:, 1] = 1 - label[:, 1]  # 坐标变换:cx=w-cx
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
            # 合并图片，坐标变为合并后的真实坐标(cx,cy,w,h)
            if i == 0:  # 左上
                x_add, y_add = min(x_center, w), min(y_center, h)
                image_merge[y_center - y_add:y_center, x_center - x_add:x_center] = image[h - y_add:h, w - x_add:w]
                label[:, 1] = label[:, 1] * w + x_center - w  # cx
                label[:, 2] = label[:, 2] * h + y_center - h  # cy
                label[:, 3:5] = label[:, 3:5] * (w, h)  # w,h
            elif i == 1:  # 右上
                x_add, y_add = min(self.input_size - x_center, w), min(y_center, h)
                image_merge[y_center - y_add:y_center, x_center:x_center + x_add] = image[h - y_add:h, 0:x_add]
                label[:, 1] = label[:, 1] * w + x_center  # cx
                label[:, 2] = label[:, 2] * h + y_center - h  # cy
                label[:, 3:5] = label[:, 3:5] * (w, h)  # w,h
            elif i == 2:  # 右下
                x_add, y_add = min(self.input_size - x_center, w), min(self.input_size - y_center, h)
                image_merge[y_center:y_center + y_add, x_center:x_center + x_add] = image[0:y_add, 0:x_add]
                label[:, 1] = label[:, 1] * w + x_center  # cx
                label[:, 2] = label[:, 2] * h + y_center  # cy
                label[:, 3:5] = label[:, 3:5] * (w, h)  # w,h
            else:  # 左下
                x_add, y_add = min(x_center, w), min(self.input_size - y_center, h)
                image_merge[y_center:y_center + y_add, x_center - x_add:x_center] = image[0:y_add, w - x_add:w]
                label[:, 1] = label[:, 1] * w + x_center - w  # cx
                label[:, 2] = label[:, 2] * h + y_center  # cy
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
        frame_all[:, 0:2] = frame_all[:, 0:2] + frame_all[:, 2:4] / 2  # 真实坐标(cx,cy,w,h)
        screen_list = np.where((frame_all[:, 2] > self.mosaic_screen) & (frame_all[:, 3] > self.mosaic_screen),
                               True, False)  # w,h不能小于screen
        frame_all = frame_all[screen_list]
        cls_all = cls_all[screen_list]
        if len(frame_all) == 0:  # 没有标签
            frame_all = None
        return image_merge, frame_all, cls_all

    def _resize(self, image, frame):  # 将图片四周填充变为正方形，frame输入输出都为[[cx,cy,w,h]...](相对原图片的比例值)
        shape = image.shape
        w0 = shape[1]
        h0 = shape[0]
        if w0 == h0 == self.input_size:  # 不需要变形
            frame *= self.input_size
            return image, frame
        else:
            image_resize = np.full((self.input_size, self.input_size, 3), 128)
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

    def draw_image(self, image, frame_all, save_path='draw.jpg'):  # 画图(cx,cy,w,h)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        frame_all[:, 0:2] = frame_all[:, 0:2] - frame_all[:, 2:4] / 2
        frame_all[:, 2:4] = frame_all[:, 0:2] + frame_all[:, 2:4]  # (x_min,y_min,x_max,y_max)
        for frame in frame_all:
            x1, y1, x2, y2 = frame
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255, 0), thickness=2)
        cv2.imwrite(save_path, image)
