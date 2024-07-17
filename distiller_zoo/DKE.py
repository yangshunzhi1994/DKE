import torch
import torch.nn as nn
import torch.nn.functional as F


def KL_divergence(student_logit, teacher_logit, T):
    KD_loss = nn.KLDivLoss(reduction='none')(F.log_softmax(student_logit / T, dim=1), F.softmax(teacher_logit / T, dim=1)) * T * T
    return KD_loss.sum(1)


def compute_similarity(logit_hd, logit_enh):
    logit_hd_norm = F.normalize(logit_hd, p=2, dim=1)
    logit_enh_norm = F.normalize(logit_enh, p=2, dim=1)
    similarity_matrix = torch.mm(logit_hd_norm, logit_enh_norm.t())

    squared_diff = (logit_hd_norm.unsqueeze(1) - logit_enh_norm.unsqueeze(0)) ** 2
    distance_matrix = torch.sqrt(torch.sum(squared_diff, dim=2) + 1e-8)

    return similarity_matrix, distance_matrix


def weighting(logit, T):
    B_size = logit.shape[0]
    vector = logit.view(-1)
    weighting = F.softmax(vector/T + 1e-8, dim=0) * len(vector)
    return weighting.view(B_size, -1)


class DKE(nn.Module):
    def __init__(self, T, gamma, alpha, beta, multiple):
        super(DKE, self).__init__()
        self.T = T
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.multiple = multiple

    def forward(self, logit_s, logit_s_aux, logit_t, logit_t_aux, targets):

        if logit_s_aux == None and logit_t_aux == None:
            loss_cls = F.cross_entropy(logit_s, targets)
            n_samples = int(len(logit_s) / 2)
            # Difference size of teacher and student networks in HD samples
            loss_KD = KL_divergence(logit_s, logit_t, self.T).mean()
            teacher_similarity, teacher_difference = compute_similarity(logit_t[:n_samples], logit_t[n_samples:])
            student_similarity, student_difference = compute_similarity(logit_s[:n_samples], logit_s[n_samples:])
            relationship_loss = weighting(-teacher_similarity, self.T) * F.mse_loss(student_similarity, teacher_similarity, reduction='none') + \
                                weighting(teacher_difference, self.T) * F.mse_loss(student_difference, teacher_difference, reduction='none')
            relationship_loss = relationship_loss.mean()
        else:
            logit_s_aux = torch.cat(logit_s_aux, dim=0)
            logit_t_aux = torch.cat(logit_t_aux, dim=0)
            num_aux = int(logit_s_aux.shape[0] // targets.shape[0])

            labels = torch.stack([targets * num_aux + i for i in range(num_aux)], 1).view(-1)
            loss_cls = F.cross_entropy(logit_s, targets) + F.cross_entropy(logit_s_aux, labels)

            n_samples = int(len(logit_s) / 2)
            # Difference size of teacher and student networks in HD samples
            loss_KD = KL_divergence(logit_s, logit_t, self.T).mean() + KL_divergence(logit_s_aux, logit_t_aux, self.T).mean()

            teacher_similarity, teacher_difference = compute_similarity(logit_t[:n_samples], logit_t[n_samples:])
            student_similarity, student_difference = compute_similarity(logit_s[:n_samples], logit_s[n_samples:])
            teacher_similarity_aux, teacher_difference_aux = compute_similarity(logit_t_aux[:n_samples], logit_t_aux[n_samples:])
            student_similarity_aux, student_difference_aux = compute_similarity(logit_s_aux[:n_samples], logit_s_aux[n_samples:])

            relationship_loss = weighting(-teacher_similarity, self.T) * F.mse_loss(student_similarity, teacher_similarity, reduction='none') + \
                                weighting(teacher_difference, self.T) * F.mse_loss(student_difference, teacher_difference, reduction='none')
            relationship_loss_aux = weighting(-teacher_similarity_aux, self.T) * F.mse_loss(student_similarity_aux, teacher_similarity_aux, reduction='none') + \
                                weighting(teacher_difference_aux, self.T) * F.mse_loss(student_difference_aux, teacher_difference_aux, reduction='none')
            relationship_loss = relationship_loss.mean() + relationship_loss_aux.mean()

        return self.gamma * loss_cls + self.alpha * loss_KD + self.beta * relationship_loss
