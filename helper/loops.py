from __future__ import print_function, division

import sys
import time
import torch
import torch.nn.functional as F
from .util import AverageMeter, accuracy
from distiller_zoo import hcl


def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target, index) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output = model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

    return top1.avg, losses.avg


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        data_time.update(time.time() - end)

        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target, index = data
        input = input.float()
        if torch.cuda.is_available():
            input, target, index = input.cuda(), target.cuda(), index.cuda()
            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()
        # ===================forward=====================
        preact = False
        if opt.distill in ['abound']:
            preact = True

        if opt.use_DA:
            if opt.multiple == "4X":
                input = torch.cat((input[:, 0], input[:, 2], input[:, 1], input[:, 3]), dim=0)
                target = torch.cat((target[:, 0], target[:, 2], target[:, 1], target[:, 3]), dim=0).squeeze(dim=1)
                index = torch.cat((index, index, index, index), dim=0)
            else:
                input = torch.cat((input[:,0], input[:,1]), dim=0)
                target = torch.cat((target[:, 0], target[:, 1]), dim=0).squeeze(dim=1)
                index = torch.cat((index, index), dim=0)
            ############### Note that our work does not involve feature distillation ##############
            if opt.dataset == 'cifar100':
                logit_s, logit_s_aux = model_s(input, is_feat=False, preact=preact)
                logit_t, logit_t_aux = model_t(input, is_feat=False, preact=preact)
            else:
                logit_s = model_s(input, is_feat=False, preact=preact)
                logit_t = model_t(input, is_feat=False, preact=preact)
        else:
            feat_s, logit_s = model_s(input, is_feat=True, preact=preact)
            with torch.no_grad():
                feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
                if opt.dataset == 'cifar100':
                    feat_t = [f.detach() for f in feat_t]

        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'edk' or opt.distill == 'ccd':
            if opt.use_DA:
                if opt.dataset == 'cifar100':
                    loss = criterion_kd(logit_s, logit_s_aux, logit_t, logit_t_aux, target)
                else:
                    loss = criterion_kd(logit_s, None, logit_t, None, target)
            else:
                loss = criterion_kd(logit_s, logit_t, target)
        elif opt.distill == 'atd':
            loss = criterion_kd(logit_s, logit_t, target, index)
        elif opt.distill == 'ReviewKD_atd':
            loss = opt.Review * hcl(feat_s, feat_t[1:]) + criterion_kd(logit_s, logit_t, target, index)
        elif opt.distill == 'ReviewKD':
            loss_kd = hcl(feat_s, feat_t[1:])
        elif opt.distill == 'ofd':
            loss_kd = criterion_kd(feat_s[1:-1], feat_t[1:-1])
        elif opt.distill == 'dkd':
            loss_kd = criterion_kd(logit_s, logit_t, target, opt.alpha, opt.beta, opt.kd_T)
        elif opt.distill == 'hint':
            f_s = module_list[1](feat_s[opt.hint_layer])
            f_t = feat_t[opt.hint_layer]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'attention':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'nst':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'rkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'pkt':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'kdsvd':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'correlation':
            f_s = module_list[1](feat_s[-1])
            f_t = module_list[2](feat_t[-1])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'abound':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'fsp':
            # can also add loss to this stage
            loss_kd = criterion_kd(feat_s[:-1], feat_t[:-1])
        elif opt.distill == 'factor':
            factor_s = module_list[1](feat_s[-2])
            factor_t = module_list[2](feat_t[-2], is_factor=True)
            loss_kd = criterion_kd(factor_s, factor_t)
        else:
            raise NotImplementedError(opt.distill)

        if opt.distill == 'atd' or opt.distill == 'ReviewKD_atd' or opt.distill == 'edk' or opt.distill == 'ccd':
            pass
        elif opt.distill == 'dkd':
            loss_cls = criterion_cls(logit_s, target)
            loss = opt.gamma * loss_cls + opt.alpha * loss_kd
        else:
            # cls + kl div
            loss_cls = criterion_cls(logit_s, target)
            loss_div = criterion_div(logit_s, logit_t)
            loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            if opt.use_DA and opt.dataset == 'cifar100':
                output, _ = model(input)
            else:
                output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    return top1.avg, top5.avg, losses.avg