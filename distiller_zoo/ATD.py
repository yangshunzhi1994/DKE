from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataset.imagenet import ImageFolderLMDB
from dataset.car196 import CAR196
from dataset.tinyimagenet import ImageFolderLMDB as tiny_ImageFolderLMDB

class ATD(nn.Module):
    def __init__(self, temperature, gamma, alpha, beta):
        super(ATD, self).__init__()
        self.temperature = temperature
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

    def forward(self, logit_s, logit_t, target, index):

        soft_logit_s = torch.div(logit_s, self.temperature[index].unsqueeze(1).expand(logit_s.shape))
        soft_logit_t = torch.div(logit_t, self.temperature[index].unsqueeze(1).expand(logit_t.shape))
        loss_div = nn.KLDivLoss(reduction="none")(F.log_softmax(soft_logit_s, dim=1), F.softmax(soft_logit_t, dim=1)).sum(1) \
                   * self.temperature[index] * self.temperature[index]

        loss_mixup = mixup_logit(logit_s, logit_t, self.temperature[index])
        loss = self.gamma * F.cross_entropy(logit_s, target) + self.alpha * loss_div.mean() + self.beta * loss_mixup

        return loss


def mixup_logit(logit_s, logit_t, temperature):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    batch_size = logit_s.size()[0]
    index = torch.randperm(batch_size).cuda()

    lam = temperature / temperature.max()
    lam = lam.unsqueeze(1).expand(logit_s.shape)

    mixed_logits_s = (1 - lam) * logit_s + lam * logit_s[index, :]
    mixed_logits_t = (1 - lam) * logit_t + lam * logit_t[index, :]
    loss = nn.KLDivLoss(reduction="batchmean")(F.log_softmax(mixed_logits_s, dim=1), F.softmax(mixed_logits_t, dim=1))

    return loss


















class CIFAR100Instance(datasets.CIFAR100):
    """CIFAR100Instance Dataset."""

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index

class SVHNInstance(datasets.SVHN):

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index

class CINIC10Instance(datasets.ImageFolder):

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index


class Sample_entropy(nn.Module):
    def __init__(self, dataset, kd_T, batch_size, num_workers):
        super(Sample_entropy, self).__init__()
        self.dataset = dataset
        self.kd_T = kd_T
        self.batch_size = batch_size
        self.num_workers = num_workers

    def forward(self, net):
        if self.dataset == 'cifar100':
            data_folder = '../datasets/CIFAR100/'
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
            train_set = CIFAR100Instance(root=data_folder, download=False, train=True, transform=train_transform)
        elif self.dataset == 'imagenet':
            data_folder = '../datasets/Imagenet_lmdb/train.lmdb'
            train_transform = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            train_set = ImageFolderLMDB(data_folder, transform=train_transform)
        elif self.dataset == 'CAR196':
            train_set = CAR196(split='Training')
        elif self.dataset == 'SVHN':
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            train_set = SVHNInstance(root='../datasets/SVHN/', split='train', download=False, transform=train_transform)
        elif self.dataset == 'CINIC10':
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835)),
            ])
            train_set = CINIC10Instance(root='../datasets/CINIC10/train/', transform=train_transform)
        elif self.dataset == 'TinyImageNet':
            train_set = tiny_ImageFolderLMDB('../datasets/tiny-imagenet-200/train.lmdb',
                                             transforms.Compose([
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                             ]),
                                             split='Training')
        else:
            raise NotImplementedError(self.dataset)

        train_loader = DataLoader(train_set,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers,
                                  worker_init_fn=np.random.seed(12),
                                  pin_memory=True)
        entropy_teacher = torch.zeros(len(train_set)).cuda()
        for idx, data in enumerate(train_loader):
            input, target, index = data
            input = input.float()
            input, target, index = input.cuda(), target.cuda(), index.cuda()
            with torch.no_grad():
                feat_t, logit_t = net(input, is_feat=True)
                entropy = torch.sum(-F.softmax(logit_t+1e-10, dim=1) * F.log_softmax(logit_t+1e-10, dim=1), dim=1)
                entropy_teacher[index] = entropy

        temperature = F.softmax(-entropy_teacher, dim=0) * self.kd_T * len(train_set)
        return temperature
