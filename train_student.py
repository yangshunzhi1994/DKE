"""
the general training framework
"""

from __future__ import print_function
import numpy
import os
import argparse
import time
import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

import numpy as np
from torch.utils.data import DataLoader
from models import model_dict
from models.imagenet import imagenet_model_dict
from models.util import Embed, ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from dataset.SVHN import get_SVHN_dataloaders
from dataset.CINIC10 import get_CINIC10_dataloaders
from dataset.imagenet import get_imagenet_dataloader
from dataset.car196 import CAR196
from dataset.tinyimagenet import get_tinyimagenet_dataloader

from helper.util import adjust_learning_rate, adjust_learning_rate_wram_up

from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss, OFD, DKD, DKE, CCD
from distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss, ATD, Sample_entropy, build_review_kd
from crd.criterion import CRDLoss

from helper.loops import train_distill as train, validate
from helper.pretrain import init


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')
    parser.add_argument('--wram_up', type=int, default=20, help='wram up')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet', 'CAR196', 'SVHN',
                                                                            'TinyImageNet', 'CINIC10'], help='dataset')
    parser.add_argument('--use_DA', type=eval,  choices=[True, False],  default='True', help='This is a boolean flag.')
    parser.add_argument('--multiple', type=str, default='2X', choices=['2X', '4X'], help='multiplet')

    # model
    parser.add_argument('--model_s', type=str, default='resnet20',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2', 'vgg8',
                                 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50', 'MobileNetV2', 'ShuffleV1',
                                 'ShuffleV2', 'resnet18', 'MobileNetV1', 'wrn_16_2_aux', 'wrn_40_1_aux', 'resnet20_aux',
                                 'resnet8x4_aux', 'MobileNetV2_aux', 'ShuffleV1_aux', 'ShuffleV2_aux'])
    parser.add_argument('--path_t', type=str, default='./save/models/resnet56_vanilla/ckpt_epoch_240.pth', help='teacher model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='DKE', choices=['kd', 'hint', 'attention', 'similarity', 'dkd',
                                                                      'correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                                        'rkd', 'pkt', 'abound', 'factor', 'nst', 'ofd',
                                                                        'atd', 'ReviewKD_atd', 'ReviewKD', 'DKE', 'ccd'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=1.0, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=1.0, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=None, help='weight balance for other losses')
    parser.add_argument('-Re', '--Review', type=float, default=0.0, help='ReviewKD_atd')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')
    parser.add_argument('--seed', type=int, default=0, help='seed')

    # hint layer
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])

    opt = parser.parse_args()

    if opt.dataset == 'cifar100':
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'MobileNetV2_aux', 'ShuffleV1_aux', 'ShuffleV2_aux']:
        opt.learning_rate = opt.learning_rate / 5

    # set the path according to the environment
    opt.model_path = './save/student_model'
    opt.tb_path = './save/student_tensorboards'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])

    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)
    if opt.distill == 'DKE' or opt.distill == 'ccd':
        opt.model_name = 'S({})_T({})_{}_{}_r({})_a({})_b({})_t({})_s({})_Mu({})_{}'.format(opt.model_s, opt.model_t,
                 opt.dataset, opt.distill, opt.gamma, opt.alpha, opt.beta, opt.kd_T, opt.seed, opt.multiple, opt.trial)
    else:
        opt.model_name = 'S({})_T({})_{}_{}_r({})_a({})_b({})_t({})_s({})_Re({})_{}'.format(opt.model_s, opt.model_t,
                 opt.dataset, opt.distill, opt.gamma, opt.alpha, opt.beta, opt.kd_T, opt.seed, opt.Review, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)

    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[-1] != 'aux':
        if segments[0] != 'wrn':
            return segments[0]
        else:
            return segments[0] + '_' + segments[1] + '_' + segments[2]
    else:
        return model_path.split('/')[-2]


def load_teacher(model_path, n_cls, dataset):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    if dataset == 'cifar100':
        model = model_dict[model_t](num_classes=n_cls)
        if "_aux" in model_t:
            model.load_state_dict(torch.load(model_path)['net'])
        else:
            model.load_state_dict(torch.load(model_path)['model'])
    elif dataset == 'imagenet':
        model = imagenet_model_dict[model_t](num_classes=n_cls)
        model.load_state_dict(torch.load(model_path))
    elif dataset == 'CAR196':
        model = imagenet_model_dict[model_t](num_classes=n_cls)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
    elif dataset == 'SVHN':
        model = model_dict[model_t](num_classes=n_cls)
        model.load_state_dict(torch.load(model_path)['model'])
    elif dataset == 'CINIC10':
        model = model_dict[model_t](num_classes=n_cls)
        model.load_state_dict(torch.load(model_path)['model'])
    elif dataset == 'TinyImageNet':
        model = model_dict[model_t](num_classes=n_cls)
        model.load_state_dict(torch.load(model_path)['model'])
    else:
        raise NotImplementedError(dataset)
    print('==> done')
    return model

def main():
    best_acc = 0
    opt = parse_option()
    # tensorboard logger
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # dataloader
    if opt.dataset == 'cifar100':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
        else:
            train_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers, is_instance=True,
                                                                        use_DA=opt.use_DA, multiple=opt.multiple)
        n_cls = 100
        data = torch.randn(2, 3, 32, 32)
        model_s = model_dict[opt.model_s](num_classes=n_cls)
    elif opt.dataset == 'imagenet':
        train_loader, val_loader = get_imagenet_dataloader(dataset='imagenet', batch_size=opt.batch_size,
                                                 num_workers=opt.num_workers, use_DA=opt.use_DA, multiple=opt.multiple)
        n_cls = 1000
        data = torch.randn(2, 3, 224, 224)
        model_s = imagenet_model_dict[opt.model_s](num_classes=n_cls)
    elif opt.dataset == 'CAR196':
        train_loader = DataLoader(CAR196(split='Training'), batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers,
                              worker_init_fn=np.random.seed(12), pin_memory=True)
        val_loader = DataLoader(CAR196(split='Testing'), batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)
        n_cls = 196
        data = torch.randn(2, 3, 224, 224)
        model_s = imagenet_model_dict[opt.model_s](num_classes=n_cls)
    elif opt.dataset == 'SVHN':
        train_loader, val_loader = get_SVHN_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 10
        data = torch.randn(2, 3, 32, 32)
        model_s = model_dict[opt.model_s](num_classes=n_cls)
    elif opt.dataset == 'CINIC10':
        train_loader, val_loader = get_CINIC10_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 10
        data = torch.randn(2, 3, 32, 32)
        model_s = model_dict[opt.model_s](num_classes=n_cls)
    elif opt.dataset == 'TinyImageNet':
        train_loader, val_loader = get_tinyimagenet_dataloader(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 200
        data = torch.randn(2, 3, 64, 64)
        model_s = model_dict[opt.model_s](num_classes=n_cls)
    else:
        raise NotImplementedError(opt.dataset)

    # model
    model_t = load_teacher(opt.path_t, n_cls, opt.dataset)
    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    trainable_list = nn.ModuleList([])
    if opt.distill == 'ReviewKD_atd' or opt.distill == 'ReviewKD':
        cnn = build_review_kd(opt.model_s, model_s, teacher=get_teacher_name(opt.path_t))
        module_list.append(cnn)
        trainable_list.append(cnn)
    else:
        module_list.append(model_s)
        trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'DKE':
        criterion_kd = DKE(opt.kd_T, opt.gamma, opt.alpha, opt.beta, opt.multiple)
    elif opt.distill == 'ccd':
        criterion_kd = CCD(opt.kd_T, opt.gamma, opt.alpha, opt.beta, opt.multiple)
    elif opt.distill == 'atd':
        temperature = Sample_entropy(opt.dataset, opt.kd_T, opt.batch_size, opt.num_workers).cuda()(model_t.cuda())
        criterion_kd = ATD(temperature, opt.gamma, opt.alpha, opt.beta)
    elif opt.distill == 'ReviewKD_atd':
        temperature = Sample_entropy(opt.dataset, opt.kd_T, opt.batch_size, opt.num_workers).cuda()(model_t.cuda())
        criterion_kd = ATD(temperature, opt.gamma, opt.alpha, opt.beta)
    elif opt.distill == 'ReviewKD':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'ofd':
        criterion_kd = OFD(model_t, opt.model_s, opt.batch_size)
        trainable_list.append(criterion_kd)
    elif opt.distill == 'dkd':
        criterion_kd = DKD()
    elif opt.distill == 'hint':
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        module_list.append(regress_s)
        trainable_list.append(regress_s)
    elif opt.distill == 'crd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif opt.distill == 'attention':
        criterion_kd = Attention()
    elif opt.distill == 'nst':
        criterion_kd = NSTLoss()
    elif opt.distill == 'similarity':
        criterion_kd = Similarity()
    elif opt.distill == 'rkd':
        criterion_kd = RKDLoss()
    elif opt.distill == 'pkt':
        criterion_kd = PKT()
    elif opt.distill == 'kdsvd':
        criterion_kd = KDSVD()
    elif opt.distill == 'correlation':
        criterion_kd = Correlation()
        embed_s = LinearEmbed(feat_s[-1].shape[1], opt.feat_dim)
        embed_t = LinearEmbed(feat_t[-1].shape[1], opt.feat_dim)
        module_list.append(embed_s)
        module_list.append(embed_t)
        trainable_list.append(embed_s)
        trainable_list.append(embed_t)
    elif opt.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        # add this as some parameters in VIDLoss need to be updated
        trainable_list.append(criterion_kd)
    elif opt.distill == 'abound':
        s_shapes = [f.shape for f in feat_s[1:-1]]
        t_shapes = [f.shape for f in feat_t[1:-1]]
        connector = Connector(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(connector)
        init_trainable_list.append(model_s.get_feat_modules())
        criterion_kd = ABLoss(len(feat_s[1:-1]))
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
        # classification
        module_list.append(connector)
    elif opt.distill == 'factor':
        s_shape = feat_s[-2].shape
        t_shape = feat_t[-2].shape
        paraphraser = Paraphraser(t_shape)
        translator = Translator(s_shape, t_shape)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(paraphraser)
        criterion_init = nn.MSELoss()
        init(model_s, model_t, init_trainable_list, criterion_init, train_loader, logger, opt)
        # classification
        criterion_kd = FactorTransfer()
        module_list.append(translator)
        module_list.append(paraphraser)
        trainable_list.append(translator)
    elif opt.distill == 'fsp':
        criterion_kd = FSP(feat_s[:-1], feat_t[:-1])
        trainable_list.append(criterion_kd)
        # classification training
        pass
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    module_list.cuda()
    criterion_list.cuda()
    cudnn.benchmark = True

    # routine
    for epoch in range(1, opt.epochs + 1):

        if opt.wram_up > 1:
            learning_rate = adjust_learning_rate_wram_up(epoch, opt, optimizer)
        else:
            learning_rate = adjust_learning_rate(epoch, opt, optimizer)

        f = open(opt.save_folder + '.txt', 'a')
        f.write('\n\nEpoch: %d, learning rate:  %0.8f\n' % (epoch, learning_rate))
        f.close()

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, opt)
        time2 = time.time()
        test_acc, test_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)

        f = open(opt.save_folder + '.txt', 'a')
        f.write("epoch:  %d, train_acc:  %0.2f, train_loss:  %0.2f, total time:  %0.2f\n" % (epoch, train_acc, train_loss, time2 - time1))
        f.write("epoch:  %d, test_acc_top1:  %0.2f, test_acc_top5:  %0.2f, test_loss:  %0.2f\n" % (epoch, test_acc, test_acc_top5, test_loss))
        f.close()

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)
        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_loss', test_loss, epoch)
        logger.log_value('test_acc_top5', test_acc_top5, epoch)

        # regular saving
        if epoch % opt.save_freq == 0:
            f = open(opt.save_folder + '.txt', 'a')
            f.write('==> Saving...\n')
            f.close()

            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'accuracy': test_acc,
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            torch.save(state, save_file)

            f = open(opt.save_folder + '.txt', 'a')
            f.write('saving the best model!\n')
            f.close()

    f = open(opt.save_folder + '.txt', 'a')
    f.write('best accuracy: %0.2f\n' % best_acc)
    f.close()

    # save model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
