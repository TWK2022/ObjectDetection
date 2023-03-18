import tqdm
import torch
from block.metric_get import tp_tn_fp_fn, nms_tp_fn_fp, confidence_screen, nms


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
        nms_tp_all = 0
        nms_fn_all = 0
        nms_fp_all = 0
        for item, (image_batch, true_batch, judge_batch, label_list) in enumerate(tqdm.tqdm(val_dataloader)):
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
            # 非极大值抑制前(所有输出)
            tp, tn, fp, fn = tp_tn_fp_fn(pred_batch, true_batch, judge_batch, args.confidence_threshold,
                                         args.iou_threshold)
            tp_all += tp
            tn_all += tn
            fp_all += fp
            fn_all += fn
            # 非极大值抑制后(最终显示的框)
            for i in range(len(pred_batch[0])):  # 遍历每张图片
                true = label_list[i].to(args.device)
                pred = [_[i] for _ in pred_batch]  # (Cx,Cy,w,h)
                pred = confidence_screen(pred, args.confidence_threshold)[:100]  # 置信度筛选，最多取前100
                if len(pred) == 0:  # 该图片没有预测值
                    nms_fn_all += len(true)
                    continue
                pred[:, 0:2] = pred[:, 0:2] - 1 / 2 * pred[:, 2:4]  # (x_min,y_min,w,h)真实坐标
                pred = nms(pred, args.iou_threshold)  # 非极大值抑制
                if len(true) == 0:  # 该图片没有标签
                    nms_fp_all += len(pred)
                    continue
                # nms_tp, nms_fn, nms_fp = nms_tp_fn_fp(pred, true, args.iou_threshold)
                # nms_tp_all += nms_tp
                # nms_fn_all += nms_fn
                # nms_fp_all += nms_fp
        # 计算平均损失
        val_loss = val_loss / (item + 1)
        val_frame_loss = val_frame_loss / (item + 1)
        val_confidence_loss = val_confidence_loss / (item + 1)
        val_class_loss = val_class_loss / (item + 1)
        print('\n| val_loss{:.4f} | val_frame_loss:{:.4f} | val_confidence_loss:{:.4f} |'
              ' val_class_loss:{:.4f} |'
              .format(val_loss, val_frame_loss, val_confidence_loss, val_class_loss))
        # 计算非极大值抑制前平均指标(所有输出)
        accuracy = (tp_all + tn_all) / (tp_all + tn_all + fp_all + fn_all)
        precision = tp_all / (tp_all + fp_all + 0.001)
        recall = tp_all / (tp_all + fn_all + 0.001)
        m_ap = precision * recall
        print('| accuracy:{:.4f} | precision:{:.4f} | recall:{:.4f} | m_ap:{:.4f} |'
              .format(accuracy, precision, recall, m_ap))
        # 计算非极大值抑制后平均指标(最终显示的框)
        nms_precision = nms_tp_all / (nms_tp_all + nms_fp_all + 0.001)
        nms_recall = nms_tp_all / (nms_tp_all + nms_fn_all + 0.001)
        nms_m_ap = nms_precision * nms_recall
        print('| nms_precision:{:.4f} | nms_recall:{:.4f} | nms_m_ap:{:.4f} |'
              .format(nms_precision, nms_recall, nms_m_ap))
    return val_loss, val_frame_loss, val_confidence_loss, val_class_loss, accuracy, precision, recall, m_ap, \
           nms_precision, nms_recall, nms_m_ap
