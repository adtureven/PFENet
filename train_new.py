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
    
    ### multi-processing training is deprecated
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False    # sync_bn is deprecated 
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

    for param in model.layer0.parameters():
        param.requires_grad = False
    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = False
    for param in model.layer3.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = False
        
    optimizer_pfenet = torch.optim.SGD(
        [
        {'params': model.down_query.parameters()},
        {'params': model.down_supp.parameters()},
        {'params': model.init_merge.parameters()},
        {'params': model.alpha_conv.parameters()},
        {'params': model.beta_conv.parameters()},
        {'params': model.inner_cls.parameters()},
        {'params': model.res1.parameters()},
        {'params': model.res2.parameters()},        
        {'params': model.cls.parameters()}],
        lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay) 

    # 初始化主动学习模块
    active_learning_module = ActiveLearningModule(mid_channels=1536, high_channels=2048)  # 根据你的模型设计调整这些值
    active_learning_module = active_learning_module.cuda()
    # 冻结主动学习模块的部分层，假设你只希望训练部分层
    for param in active_learning_module.conv_mid.parameters():
        param.requires_grad = True
    for param in active_learning_module.conv_prior.parameters():
        param.requires_grad = True
    for param in active_learning_module.conv_high.parameters():
        param.requires_grad = True
    for param in active_learning_module.final_conv.parameters():
        param.requires_grad = True
    # 定义优化器
    optimizer_alm = torch.optim.Adam(
        active_learning_module.parameters(),
        lr=args.base_lr, weight_decay=args.weight_decay
    )

    global logger, writer
    logger = get_logger()
    writer = SummaryWriter(args.save_path)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))
    logger.info(model)
    print(args)

    # model = torch.nn.DataParallel(model.cuda())
    model = model.cuda()

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

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer_pfenet.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))


    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    assert args.split in [0, 1, 2, 3, 999]
    train_transform = [
        transform.RandScale([args.scale_min, args.scale_max]),
        transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.padding_label),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=args.padding_label),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)]
    train_transform = transform.Compose(train_transform)
    train_data = dataset.SemData(split=args.split, shot=21, data_root=args.data_root, \
                                data_list=args.train_list, transform=train_transform, mode='train', \
                                use_coco=args.use_coco, use_split_coco=args.use_split_coco)

    # # 随机选择20个支持样本作为候选子集
    # train_data_candidate_sup = dataset.SemData(split=args.split, shot=20, data_root=args.data_root, \
    #                             data_list=args.train_list, transform=train_transform, mode='train', \
    #                             use_coco=args.use_coco, use_split_coco=args.use_split_coco)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    # train_loader_candidate_sup = torch.utils.data.DataLoader(train_data_candidate_sup, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    if args.evaluate:
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

    max_iou = 0.
    filename = 'PFENet.pth'

    for epoch in range(args.start_epoch, args.epochs):
        if args.fix_random_seed_val:
            torch.cuda.manual_seed(args.manual_seed + epoch)
            np.random.seed(args.manual_seed + epoch)
            torch.manual_seed(args.manual_seed + epoch)
            torch.cuda.manual_seed_all(args.manual_seed + epoch)
            random.seed(args.manual_seed + epoch)

        epoch_log = epoch + 1
        # # for debug
        # loss_val, mIoU_val, mAcc_val, allAcc_val, class_miou = validate(val_loader, model, active_learning_module, criterion)

        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model,active_learning_module, optimizer_pfenet, optimizer_alm,epoch)
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            writer.add_scalar('allAcc_train', allAcc_train, epoch_log)     

        if args.evaluate and (epoch % 2 == 0 or (args.epochs<=50 and epoch%1==0)):
            loss_val, mIoU_val, mAcc_val, allAcc_val, class_miou = validate(val_loader, model, active_learning_module, criterion)
            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('class_miou_val', class_miou, epoch_log)
                writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
            if class_miou > max_iou:
                max_iou = class_miou
                if os.path.exists(filename):
                    os.remove(filename)            
                filename = args.save_path + '/train_epoch_' + str(epoch) + '_'+str(max_iou)+'.pth'
                logger.info('Saving checkpoint to: ' + filename)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_pfenet_state_dict': optimizer_pfenet.state_dict(),
                    'alm_state_dict': active_learning_module.state_dict(),
                    'optimizer_alm_state_dict': optimizer_alm.state_dict()
                }, filename)

    filename = args.save_path + '/final.pth'
    logger.info('Saving checkpoint to: ' + filename)
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_pfenet_state_dict': optimizer_pfenet.state_dict(),
        'alm_state_dict': active_learning_module.state_dict(),
        'optimizer_alm_state_dict': optimizer_alm.state_dict()
    }, filename)                

    # 保存 ALM 模型
    alm_filename = args.save_path + '/active_learning_module.pth'
    logger.info('Saving ALM model to: ' + alm_filename)
    torch.save(active_learning_module.state_dict(), alm_filename)

    # 保存 PFENet 模型
    pfenet_filename = args.save_path + '/pfenet_model.pth'
    logger.info('Saving PFENet model to: ' + pfenet_filename)
    torch.save(model.state_dict(), pfenet_filename)


def train(train_loader, model, active_learning_module, optimizer_pfenet, optimizer_alm, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    active_learning_module.train()

    end = time.time()
    max_iter = args.epochs * len(train_loader)
    vis_key = 0
    print('Warmup: {}'.format(args.warmup))
    # for i, (input, target, s_input, s_mask, subcls) in enumerate(train_loader):
    for i, (input, target, s_input, s_mask, subcls) in enumerate(train_loader):
        data_time.update(time.time() - end)
        current_iter = epoch * len(train_loader) + i + 1
        index_split = -1
        if args.base_lr > 1e-6:
            poly_learning_rate(optimizer_pfenet, args.base_lr, current_iter, max_iter, power=args.power, index_split=index_split, warmup=args.warmup, warmup_step=len(train_loader)//2)
        s_input = s_input.cuda(non_blocking=True)
        s_mask = s_mask.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        if epoch < args.alm_start_epoch:
            s_input = s_input[:,0:5]
            s_mask = s_mask[:,0:5]
        else:
            s_input_1_shot = s_input[:,0:1]
            s_mask_1_shot = s_mask[:,0:1]
            s_input_20_shot = s_input[:,1:21]
            s_mask_20_shot = s_mask[:,1:21]
            # 初始化存储二范数平方的列表
            norm_squared_list = []
            kl_div_list = []    
            for i in range(20):
                torch.autograd.set_detect_anomaly(True)
                F_supp_mid, F_query_mid, Prior_mask_supp, Prior_mask_query, F_supp_high, F_query_high = \
                    extract_features(model, input, s_input_1_shot, s_mask_1_shot, s_input_20_shot[:,i:i+1])
                # 计算 ALM 的得分, score.shape = [batch_size, height, width]
                score = active_learning_module(F_supp_mid, F_query_mid, Prior_mask_supp, Prior_mask_query, F_supp_high, F_query_high)
                # 添加通道维度，变为 [4, 1, 60, 60]
                score = score.unsqueeze(1)  # 现在 score.shape = [4, 1, 60, 60]

                # 使用 F.interpolate 进行上采样，变为 [4, 1, 473, 473]
                score = F.interpolate(score, size=(473, 473), mode='bilinear', align_corners=True)

                # 如果需要去掉通道维度，可以使用 squeeze
                score = score.squeeze(1)  # 现在 score.shape = [4, 473, 473]

                # out.shape = [batch_size, num_classes, height, width]
                out, _, _, _ = model(s_x=s_input_20_shot[:,i:i+1], s_y=s_mask_20_shot[:,i:i+1], x=input, y=target)    # out.shape = [batch_size, num_classes, height, width]
                # out_softmax.shape = [batch_size, height, width]
                out_softmax = F.softmax(out, dim=1) # softmax在类别维度（dim=1）上进行[batch_size, 2,height, width]

                # 将 score 转换为概率分布
                score_prob = torch.stack([torch.sub(1, score).clone(), score.clone()], dim=1)  # [batch_size, 2, h, w]
                # # 计算 KL 散度
                # kl_div = F.kl_div(out_softmax.log(), score_prob, reduction='batchmean')
                # print("KL Divergence:", kl_div.item())
                norm_squared = torch.sum((score_prob - out_softmax) ** 2)
                # pdb.set_trace()
                norm_squared_list.append(norm_squared)
                # kl_div_list.append(kl_div)
            # 将标量转换为一维张量
            norm_squared_list = [t.unsqueeze(0).clone() for t in norm_squared_list]
            norm_squared_tensor = torch.cat(norm_squared_list, dim=0)
            # kl_div_tensor = torch.cat(kl_div_list, dim=0)
            # topk_indices = sorted(range(len(kl_div_list)), key=lambda i: kl_div_list[i])[:4]

            # 找到最小的4个二范数平方
            _, topk_indices = torch.topk(norm_squared_tensor, 4, largest=False)  # 获取最小的4个
            # _, topk_indices = torch.topk(kl_div_tensor, 4, largest=False)  # 获取最小的4个
            # 根据topk_indices获取对应的样本和标签
            topk_samples = s_input_20_shot[:, topk_indices]  # 选出最小二范数的样本
            topk_labels = s_mask_20_shot[:, topk_indices]   # 选出对应标签
            # 将 1-shot 样本和标签与选出的 topk 样本和标签合并
            s_input = torch.cat([s_input_1_shot.clone(), topk_samples.clone()], dim=1)  # [batch_size, 5, C]，合并 1-shot 样本和 topk 样本
            s_mask = torch.cat([s_mask_1_shot.clone(), topk_labels.clone()], dim=1)  # [batch_size, 5, H, W]，合并 1-shot 标签和 topk 标签
            
            
        out, output, main_loss, aux_loss = model(s_x=s_input, s_y=s_mask, x=input, y=target)    # out.shape = [batch_size, num_classes, height, width]

        if not args.multiprocessing_distributed:
            main_loss, aux_loss = torch.mean(main_loss), torch.mean(aux_loss)
        loss = main_loss + args.aux_weight * aux_loss

        if epoch >= args.alm_start_epoch:
            if epoch % 2 == 0:
                # optimizer_pfenet.zero_grad()
                # loss.backward() 
                # optimizer_pfenet.step()

                optimizer_alm.zero_grad()
                alm_loss = norm_squared_tensor.mean()  # 取均值，或者根据需要加权
                # alm_loss = kl_div_tensor.mean() 
                # alm_loss = torch.stack(kl_div_list).mean()  # 计算KL散度的均值作为总损失
                alm_loss.backward()
                optimizer_alm.step()
            else:
                optimizer_pfenet.zero_grad()
                loss.backward()
                optimizer_pfenet.step()

        else:
            optimizer_pfenet.zero_grad()
            loss.backward()
            optimizer_pfenet.step()
        
        n = input.size(0)
        if args.multiprocessing_distributed:
            main_loss, aux_loss, loss = main_loss.detach() * n, aux_loss * n, loss * n 
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(main_loss), dist.all_reduce(aux_loss), dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            main_loss, aux_loss, loss = main_loss / n, aux_loss / n, loss / n

        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
        
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        main_loss_meter.update(main_loss.item(), n)
        aux_loss_meter.update(aux_loss.item(), n)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'MainLoss {main_loss_meter.val:.4f} '
                        'AuxLoss {aux_loss_meter.val:.4f} '                        
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time,
                                                          data_time=data_time,
                                                          remain_time=remain_time,
                                                          main_loss_meter=main_loss_meter,
                                                          aux_loss_meter=aux_loss_meter,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))
        if main_process():
            writer.add_scalar('loss_train_batch', main_loss_meter.val, current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    if main_process():
        logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch, args.epochs, mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))        
    return main_loss_meter.avg, mIoU, mAcc, allAcc

def validate(val_loader, model, active_learning_module, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    # pdb.set_trace()
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
            print(i, len(val_loader))
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
                    # 添加通道维度，变为 [4, 1, 60, 60]
                    score = score.unsqueeze(1)  # 现在 score.shape = [4, 1, 60, 60]

                    # 使用 F.interpolate 进行上采样，变为 [4, 1, 473, 473]
                    score = F.interpolate(score, size=(473, 473), mode='bilinear', align_corners=True)

                    # 如果需要去掉通道维度，可以使用 squeeze
                    score = score.squeeze(1)  # 现在 score.shape = [4, 473, 473]
                    out = model(s_x=s_input_20_shot[:,i:i+1], s_y=s_mask_20_shot[:,i:i+1], x=input, y=target)    # out.shape = [batch_size, num_classes, height, width]
                    # out_softmax.shape = [batch_size, height, width]
                    out_softmax = F.softmax(out, dim=1) # softmax在类别维度（dim=1）上进行[batch_size, height, width]
                    score_prob = torch.stack([1 - score, score], dim=1)  # [batch_size, 2, h, w]
                    norm_squared = torch.sum((score_prob - out_softmax) ** 2)
                    norm_squared_list.append(norm_squared)
                # 将标量转换为一维张量
                norm_squared_list = [t.unsqueeze(0) for t in norm_squared_list]
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
