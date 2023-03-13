import tqdm
import torch
from block.metric_get import acc_pre_rec_ap


def val_get(args, val_dataloader, model, loss):
    with torch.no_grad():
        model.eval()
        val_loss = 0  # 记录验证损失
        val_frame_loss = 0  # 记录边框损失
        val_confidence_loss = 0  # 记录置信度框损失
        val_class_loss = 0  # 记录类别损失
        tp_all = 0
        tn_all = 0
        fp_all = 0
        fn_all = 0
        for item, (image_batch, true_batch, judge_batch) in enumerate(tqdm.tqdm(val_dataloader)):
            image_batch = image_batch.to(args.device, non_blocking=args.latch)  # 将输入数据放到设备上
            for i in range(len(true_batch)):  # 将标签矩阵放到对应设备上
                true_batch[i] = true_batch[i].to(args.device, non_blocking=args.latch)
            pred_batch = model(image_batch)
            # 计算损失
            loss_batch, frame_loss, confidence_loss, class_loss = loss(pred_batch, true_batch, judge_batch)
            val_loss += loss_batch.item()
            val_frame_loss += frame_loss.item()
            val_confidence_loss += confidence_loss.item()
            val_class_loss += class_loss.item()
            # 计算指标
            tp, tn, fp, fn = acc_pre_rec_ap(pred_batch, true_batch, judge_batch, args.confidence_threshold,
                                            args.iou_threshold)
            tp_all += tp
            tn_all += tn
            fp_all += fp
            fn_all += fn
        # 计算平均损失
        val_loss = val_loss / (item + 1)
        val_frame_loss = val_frame_loss / (item + 1)
        val_confidence_loss = val_confidence_loss / (item + 1)
        val_class_loss = val_class_loss / (item + 1)
        print('\n| val_loss{:.4f} | val_frame_loss:{:.4f} | val_confidence_loss:{:.4f} |'
              ' val_class_loss:{:.4f} |'
              .format(val_loss, val_frame_loss, val_confidence_loss, val_class_loss))
        # 计算平均指标
        accuracy = (tp_all + tn_all) / (tp_all + tn_all + fp_all + fn_all)
        precision = tp_all / (tp_all + fp_all + 0.001)
        recall = tp_all / (tp_all + fn_all + 0.001)
        m_ap = precision * recall
        print('| accuracy:{:.4f} | precision:{:.4f} | recall:{:.4f} | m_ap:{:.4f} |'
              .format(accuracy, precision, recall, m_ap))
    return val_loss, val_frame_loss, val_confidence_loss, val_class_loss, accuracy, precision, recall, m_ap
