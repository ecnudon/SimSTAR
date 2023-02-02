import torch
import torch.nn as nn


class basic_loss(nn.Module):
    def __init__(self):
        super(basic_loss, self).__init__()
        self.loss_term = nn.BCELoss(reduction='sum')
        self.loss_tri = nn.BCELoss(reduction='sum')

    def forward(self, term_pred, term_label, tri_pred, tri_label):
        seq_len = term_pred.size(1)
        term_loss = self.loss_term(term_pred, term_label) / seq_len
        tri_loss = self.loss_tri(tri_pred, tri_label) / seq_len
        loss = term_loss + tri_loss
        return loss


class BCEFocalLoss(nn.Module):
    def __init__(self,gamma=2, alpha=0.25, tri_contribution = 1, reduction='sum'):
        '''
        :param gamma:   if > 0, mean to focus on the prediction with more Entropy
        :param alpha: if < 0.5, mean to impair the positive label's weight
        :param tri_contribution: if > 1, means in a loss how is the contribution ratio for tri:term
        :param reduction:
        '''
        super(BCEFocalLoss, self).__init__()
        self.loss_term = FocalLoss(gamma=gamma, alpha=alpha, reduction=reduction)
        self.loss_senti = FocalLoss(gamma=gamma, alpha=alpha, reduction=reduction)
        self.tri_contribution = tri_contribution

    def forward(self, term_pred, term_label, senti_pred, senti_label):
        seq_len = term_pred.size(1)
        term_loss = self.loss_term(term_pred, term_label) / seq_len
        senti_loss = self.loss_senti(senti_pred, senti_label) / seq_len
        loss = (term_loss + senti_loss * self.tri_contribution) / self.tri_contribution
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='sum'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        predict = torch.clamp(predict,min=0.0001,max=0.9999)
        loss = - self.alpha * (1 - predict) ** self.gamma * target * torch.log(predict)\
               - (1 - self.alpha) * predict ** self.gamma * (1 - target) * torch.log(1 - predict)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss
