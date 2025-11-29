from torch.nn import BCELoss
import torch

def dice_coef_loss(inputs, target, smooth = 1e-6):
    intersection = 2.0 * (target*inputs).sum() + smooth
    union = target.sum() + inputs.sum() + smooth
    return 1 - (intersection/union)

def focal_loss(inputs, target, alpha=0.8, gamma=2):
    bce_loss = BCELoss()(inputs, target)
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    return focal_loss

def bce_dice_loss(inputs, target):
    dice_score = dice_coef_loss(inputs, target)
    bce_loss = BCELoss()
    bce_score = bce_loss(inputs, target)
    
    return bce_score + dice_score

def focal_dice_loss(inputs, target, alpha=0.8, gamma=2):
    dice_score = dice_coef_loss(inputs, target)
    focal_score = focal_loss(inputs, target, alpha, gamma)
    return focal_score + dice_score