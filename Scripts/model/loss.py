from torch.nn import BCELoss


def dice_coef_loss(inputs, target, smooth = 1e-6):
    intersection = 2.0 * (target*inputs).sum() + smooth
    union = target.sum() + inputs.sum() + smooth
    return 1 - (intersection/union)

def bce_dice_loss(inputs, target):
    dice_score = dice_coef_loss(inputs, target)
    bce_loss = BCELoss()
    bce_score = bce_loss(inputs, target)
    
    return bce_score + dice_score