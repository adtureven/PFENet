import os
import random
import time
import cv2
import numpy as np
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter

from model.PFENet import PFENet   
from util import dataset
from util import transform, config
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU

from model.ActiveLearningModule import ActiveLearningModule
from model.ActiveLearningModule import extract_features
import pdb
import itertools

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/ade20k/ade20k_pspnet50.yaml', help='config file')
    parser.add_argument('opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss

    BatchNorm = nn.BatchNorm2d

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    model = PFENet(layers=args.layers, classes=2, zoom_factor=8, \
        criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=BatchNorm, \
        pretrained=True, shot=args.shot, ppm_scales=args.ppm_scales, vgg=args.vgg)

    # 初始化主动学习模块
    active_learning_module = ActiveLearningModule(mid_channels=1536, high_channels=2048)  # 根据你的模型设计调整这些值
    active_learning_module = active_learning_module.cuda()

    global logger, writer
    logger = get_logger()
    writer = SummaryWriter(args.save_path)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))
    logger.info(model)
    print(args)

    model = torch.nn.DataParallel(model.cuda())

    if args.weight:
        if os.path.isfile(args.weight):
            logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    # 加载 ALM 权重
    if args.alm_weight:  # 假设你在命令行参数中添加了 --alm_weight
        if os.path.isfile(args.alm_weight):
            logger.info("=> loading ALM weight '{}'".format(args.alm_weight))
            alm_checkpoint = torch.load(args.alm_weight)
            active_learning_module.load_state_dict(alm_checkpoint['state_dict'])
            logger.info("=> loaded ALM weight '{}'".format(args.alm_weight))
        else:
            logger.info("=> no ALM weight found at '{}'".format(args.alm_weight))


    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    assert args.split in [0, 1, 2, 3, 999]

    if args.resized_val:
        val_transform = transform.Compose([
            transform.Resize(size=args.val_size),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])    
    else:
        val_transform = transform.Compose([
            transform.test_Resize(size=args.val_size),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])           
    val_data = dataset.SemData(split=args.split, shot=21, data_root=args.data_root, \
                            data_list=args.val_list, transform=val_transform, mode='val', \
                            use_coco=args.use_coco, use_split_coco=args.use_split_coco)
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    loss_val, mIoU_val, mAcc_val, allAcc_val, class_miou = validate(val_loader, model, criterion) 


def validate(val_loader, model, active_learning_module, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    if args.use_coco:
        split_gap = 20
    else:
        split_gap = 5
    class_intersection_meter = [0] * split_gap
    class_union_meter = [0] * split_gap

    if args.manual_seed is not None and args.fix_random_seed_val:
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)

    model.eval()
    if active_learning_module is not None:
        active_learning_module.eval()  # 设置 ALM 模块为评估模式

    end = time.time()
    if args.split != 999:
        test_num = 20000 if args.use_coco else 5000
    else:
        test_num = len(val_loader)
    assert test_num % args.batch_size_val == 0

    iter_num = 0
    total_time = 0

    for e in range(10): 
        for i, (input, target, s_input, s_mask, subcls, ori_label) in enumerate(val_loader):
            if (iter_num - 1) * args.batch_size_val >= test_num:
                break
            iter_num += 1

            # 数据加载时间
            data_time.update(time.time() - end)
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            ori_label = ori_label.cuda(non_blocking=True)
            s_input = s_input.cuda(non_blocking=True)
            s_mask = s_mask.cuda(non_blocking=True)

            # 使用 ALM 模块选择支持样本
            if active_learning_module is not None:
                s_input_1_shot = s_input[:, 0:1]
                s_mask_1_shot = s_mask[:, 0:1]
                s_input_20_shot = s_input[:, 1:21]
                s_mask_20_shot = s_mask[:, 1:21]

                norm_squared_list = []
                for i in range(20):
                    F_supp_mid, F_query_mid, Prior_mask_supp, Prior_mask_query, F_supp_high, F_query_high = \
                        extract_features(model, input, s_input_1_shot, s_mask_1_shot, s_input_20_shot[:,i:i+1])
                    # 计算 ALM 的得分, score.shape = [batch_size, height, width]
                    score = active_learning_module(F_supp_mid, F_query_mid, Prior_mask_supp, Prior_mask_query, F_supp_high, F_query_high)
                    # out.shape = [batch_size, num_classes, height, width]
                    out = model(s_x=s_input_20_shot[:,i:i+1], s_y=s_mask_20_shot[:,i:i+1], x=input, y=target)    # out.shape = [batch_size, num_classes, height, width]
                    # out_softmax.shape = [batch_size, height, width]
                    out_softmax = F.softmax(out, dim=1) # softmax在类别维度（dim=1）上进行[batch_size, height, width]
                    norm_squared = torch.sum((score - out_softmax) ** 2)
                    norm_squared_list.append(norm_squared)
                norm_squared_tensor = torch.cat(norm_squared_list, dim=0)

                # 找到最小的4个二范数平方
                _, topk_indices = torch.topk(norm_squared_tensor, 4, largest=False)  # 获取最小的4个

                # 根据topk_indices获取对应的样本和标签
                topk_samples = s_input_20_shot[:, topk_indices]  # 选出最小二范数的样本
                topk_labels = s_mask_20_shot[:, topk_indices]   # 选出对应标签
                # 将 1-shot 样本和标签与选出的 topk 样本和标签合并
                s_input = torch.cat([s_input_1_shot, topk_samples], dim=1)  # [batch_size, 5, C]，合并 1-shot 样本和 topk 样本
                s_mask = torch.cat([s_mask_1_shot, topk_labels], dim=1)  # [batch_size, 5, H, W]，合并 1-shot 标签和 topk 标签
                
            start_time = time.time()
            output = model(s_x=s_input, s_y=s_mask, x=input, y=target)
            total_time += 1
            model_time.update(time.time() - start_time)

            if args.ori_resize:
                longerside = max(ori_label.size(1), ori_label.size(2))
                backmask = torch.ones(ori_label.size(0), longerside, longerside).cuda() * 255
                backmask[0, :ori_label.size(1), :ori_label.size(2)] = ori_label
                target = backmask.clone().long()

            output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)

            # 计算损失
            loss = criterion(output, target)
            loss = torch.mean(loss)

            # 计算交并比 (IoU)
            output = output.max(1)[1]
            intersection, union, new_target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
            intersection, union, target, new_target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy(), new_target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(new_target)

            # 更新类别交并比
            subcls = subcls[0].cpu().numpy()[0]
            class_intersection_meter[(subcls - 1) % split_gap] += intersection[1]
            class_union_meter[(subcls - 1) % split_gap] += union[1]

            # 计算精度
            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            loss_meter.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            # 打印测试进度
            if ((i + 1) % (test_num / 100) == 0) and main_process():
                logger.info('Test: [{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                            'Accuracy {accuracy:.4f}.'.format(iter_num * args.batch_size_val, test_num,
                                                              data_time=data_time,
                                                              batch_time=batch_time,
                                                              loss_meter=loss_meter,
                                                              accuracy=accuracy))

    # 计算测试结果
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    # 计算类别 IoU
    class_iou_class = []
    class_miou = 0
    for i in range(len(class_intersection_meter)):
        class_iou = class_intersection_meter[i] / (class_union_meter[i] + 1e-10)
        class_iou_class.append(class_iou)
        class_miou += class_iou
    class_miou = class_miou * 1.0 / len(class_intersection_meter)

    if main_process():
        logger.info('meanIoU---Val result: mIoU {:.4f}.'.format(class_miou))
        for i in range(split_gap):
            logger.info('Class_{} Result: iou {:.4f}.'.format(i + 1, class_iou_class[i]))
        logger.info('FBIoU---Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    print('avg inference time: {:.4f}, count: {}'.format(model_time.avg, test_num))
    return loss_meter.avg, mIoU, mAcc, allAcc, class_miou


if __name__ == '__main__':
    main()
