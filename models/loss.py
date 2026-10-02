import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import softmax2d


class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        
    def forward(self, log_preds, targets):
        # targets are log-probabilities (log-softmax from heatmap generation)
        # Convert targets from log-probabilities to probabilities
        targets_prob = torch.exp(targets)
        # log_preds are logits, softmax2d(log_preds, log=True) gives log-probabilities
        log_preds_prob = softmax2d(log_preds, log=True)
        # Cross-entropy: -sum(targets_prob * log_preds_prob)
        loss = - targets_prob * log_preds_prob

        return loss.sum()