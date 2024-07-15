import torch
import torch.nn as nn
import torch.nn.functional as F


def KL_divergence(student_logit, teacher_logit, T):
    KD_loss = nn.KLDivLoss(reduction='none')(F.log_softmax(student_logit / T, dim=1), F.softmax(teacher_logit / T, dim=1)) * T * T
    return KD_loss.sum(1)


class CCD(nn.Module):
    def __init__(self, T, gamma, alpha, beta, multiple):
        super(CCD, self).__init__()
        self.T = T
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.multiple = multiple

    def forward(self, logit_s, logit_s_aux, logit_t, logit_t_aux, targets):

        if logit_s_aux == None and logit_t_aux == None:
            loss_cls = F.cross_entropy(logit_s, targets)
            n_samples = int(len(logit_s) / 2)
            cosine = F.cosine_similarity(logit_t[:n_samples], logit_t[n_samples:]) + 1
            # Difference size of teacher and student networks in HD samples
            loss_KD = KL_divergence(logit_s[:n_samples], logit_t[:n_samples], self.T) + cosine * KL_divergence(logit_s[n_samples:], logit_t[n_samples:], self.T)

        else:
            logit_s_aux = torch.cat(logit_s_aux, dim=0)
            logit_t_aux = torch.cat(logit_t_aux, dim=0)
            num_aux = int(logit_s_aux.shape[0] // targets.shape[0])

            labels = torch.stack([targets * num_aux + i for i in range(num_aux)], 1).view(-1)
            loss_cls = F.cross_entropy(logit_s, targets) + F.cross_entropy(logit_s_aux, labels)

            n_samples = int(len(logit_s) / 2)
            cosine = F.cosine_similarity(logit_t[:n_samples], logit_t[n_samples:]) + 1
            # Difference size of teacher and student networks in HD samples
            loss_KD = KL_divergence(logit_s[:n_samples], logit_t[:n_samples], self.T) + cosine * KL_divergence(logit_s[n_samples:], logit_t[n_samples:], self.T)
            loss_KD_aux = KL_divergence(logit_s_aux, logit_t_aux, self.T)

        return self.gamma * loss_cls + self.alpha * (loss_KD.mean() + loss_KD_aux.mean())