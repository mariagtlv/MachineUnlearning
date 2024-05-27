import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(CustomLoss, self).__init__()
        self.alpha = alpha 
        self.beta = beta 

    def forward(self, output, target):
        ce_loss = F.cross_entropy(output, target)

        
        penalty = torch.mean((output[:, 6] - output[:, 3]).clamp(min=0))

        loss = self.alpha * ce_loss + self.beta * penalty

        return loss