import torch
import torch.nn.functional as F

def sensitivity(preds, masks, smooth=1e-6):
    true_positive = (preds & masks).sum()
    actual_positive = masks.sum()
    sensitivity = (true_positive + smooth) / (actual_positive + smooth)
    return sensitivity

def specificity(preds, masks, smooth=1e-6):
    true_negative = ((~preds.bool()) & (~masks.bool())).sum()
    actual_negative = (~masks.bool()).sum()
    specificity = (true_negative.float() + smooth) / (actual_negative.float() + smooth)
    return specificity.item()  # return as plain number

def precision(preds, masks, smooth=1e-6):
    true_positive = (preds & masks).sum()
    predicted_positive = preds.sum()
    precision = (true_positive + smooth) / (predicted_positive + smooth)
    return precision

def f1_score(preds, masks, smooth=1e-6):
    prec = precision(preds, masks, smooth)
    sens = sensitivity(preds, masks, smooth)
    f1 = 2 * (prec * sens) / (prec + sens + smooth)
    return f1

def dice_score(pred, target, eps=1e-7):
    intersection = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = (2. * intersection + eps) / (union + eps)
    return dice.mean()

def dice_loss(pred, target):
    return  1- dice_score(pred, target)

def dice_binary(pred, target, threshold=0.5):
    pred_bin = (pred >= threshold).float()
    target_bin = (target >= threshold).float()
    return dice_score(pred_bin, target_bin)

def combined_loss(output, target, beta=0.5):
    bce = F.binary_cross_entropy_with_logits(output, target)
    dice = dice_loss(torch.sigmoid(output), target)  # assuming dice_loss calculates 1 - dice_score
    return beta * bce + (1 - beta) * dice

def pixel_accuracy(preds, targets, threshold=0.5):
    preds_bin = (preds >= threshold).float()
    correct = (preds_bin == targets).float().sum()
    total = targets.numel()
    return correct / total

def iou(preds, targets, threshold=0.5, eps=1e-7):
    preds_bin = (preds >= threshold).float()
    targets_bin = (targets >= threshold).float()
    intersection = (preds_bin * targets_bin).sum()
    total = (preds_bin + targets_bin).sum() - intersection
    return (intersection + eps) / (total + eps)