import tqdm
import torch
from model.layer import decode
from block.metric_get import confidence_screen, nms, nms_tp_fn_fp


def val_get(args, val_dataloader, model, loss, ema, data_len):
    with torch.no_grad():
        model = ema.ema if args.ema else model.eval()
        decode_model = decode(args.input_size)
        val_loss = 0  # 记录验证损失
        val_frame_loss = 0  # 记录边框损失
        val_confidence_loss = 0  # 记录置信度框损失
        val_class_loss = 0  # 记录类别损失
        nms_tp_all = 0
        nms_fp_all = 0
        nms_fn_all = 0
        tqdm_len = data_len // args.batch * args.device_number
        tqdm_show = tqdm.tqdm(total=tqdm_len)
        for index, (image_batch, true_batch, judge_batch, label_list) in enumerate(val_dataloader):
            image_batch = image_batch.to(args.device, non_blocking=args.latch)  # 将输入数据放到设备上
            for i in range(len(true_batch)):  # 将标签矩阵放到对应设备上
                true_batch[i] = true_batch[i].to(args.device, non_blocking=args.latch)
            pred_batch = model(image_batch)
            clone_batch = [_.clone() for _ in pred_batch]  # 计算损失会改变pred_batch
            # 计算损失
            loss_batch, frame_loss, confidence_loss, class_loss = loss(pred_batch, true_batch, judge_batch)
            val_loss += loss_batch.item()
            val_frame_loss += frame_loss.item()
            val_confidence_loss += confidence_loss.item()
            val_class_loss += class_loss.item()
            # 解码输出
            clone_batch = decode_model(clone_batch)  # (Cx,Cy,w,h,confidence...)原始输出->(Cx,Cy,w,h,confidence...)真实坐标
            # 统计指标
            for i in range(clone_batch[0].shape[0]):  # 遍历每张图片
                true = label_list[i].to(args.device)
                pred = [_[i] for _ in clone_batch]  # (Cx,Cy,w,h)真实坐标
                pred = confidence_screen(pred, args.confidence_threshold)  # 置信度筛选
                if len(pred) == 0:  # 该图片没有预测值
                    nms_fn_all += len(true)
                    continue
                pred[:, 0:2] = pred[:, 0:2] - pred[:, 2:4] / 2  # (x_min,y_min,w,h)真实坐标
                true[:, 0:2] = true[:, 0:2] - true[:, 2:4] / 2  # (x_min,y_min,w,h)真实坐标
                pred = nms(pred, args.iou_threshold)[:100]  # 非极大值抑制，最多100
                if len(true) == 0:  # 该图片没有标签
                    nms_fp_all += len(pred)
                    continue
                nms_tp, nms_fp, nms_fn = nms_tp_fn_fp(pred, true, args.iou_threshold)
                nms_tp_all += nms_tp
                nms_fn_all += nms_fn
                nms_fp_all += nms_fp
            # tqdm
            tqdm_show.set_postfix({'val_loss': loss_batch.item()})  # 添加显示
            tqdm_show.update(1)  # 更新进度条
        # tqdm
        tqdm_show.close()
        # 计算平均损失
        val_loss /= index + 1
        val_frame_loss /= index + 1
        val_confidence_loss /= index + 1
        val_class_loss /= index + 1
        print(f'\n| 验证 | val_loss{val_loss:.4f} | val_frame_loss:{val_frame_loss:.4f} |'
              f' val_confidence_loss:{val_confidence_loss:.4f} | val_class_loss:{val_class_loss:.4f} |')
        # 计算指标
        precision = nms_tp_all / (nms_tp_all + nms_fp_all + 0.001)
        recall = nms_tp_all / (nms_tp_all + nms_fn_all + 0.001)
        m_ap = precision * recall
        print('| 验证 | precision:{:.4f} | recall:{:.4f} | m_ap:{:.4f} |'.format(precision, recall, m_ap))
    return val_loss, val_frame_loss, val_confidence_loss, val_class_loss, precision, recall, m_ap
