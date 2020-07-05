import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, age, gender, race):
        age_pred = output[:, :13]
        age_pred = torch.sum(age_pred, 1)
        gender_pred = output[:, 13: 15]
        race_pred = output[:, 15:]
        age_loss = F.smooth_l1_loss(age_pred.view(-1, 1), age.float().cuda())
        gender_loss = F.cross_entropy(gender_pred, torch.flatten(gender).cuda(), reduction='sum')
        race_loss = F.cross_entropy(race_pred, torch.flatten(race).cuda(), reduction='sum')
        return age_loss, gender_loss, race_loss

