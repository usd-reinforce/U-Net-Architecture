import torch
import torch.nn.functional as F

def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float=1e-6) -> torch.Tensor:
    """
    computes the dice loss for binary segmentation.
    :param pred: Tensor of model predictions (B, in_c, H, W).
    :param target: Tensor of ground truth (B, in_c, H, W).
    :param smooth: smoothing factor to avoid division by zero.
    :return: torch.Tensor
    """

    pred = torch.sigmoid(pred)

    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)

    return 1 - dice.mean()

def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float=1e-6) -> torch.Tensor:
    pred = torch.sigmoid(pred)

    intersection = (pred * target).sum(dim=(2, 3))
    union = (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection)

    total = (intersection - smooth) / (union - smooth)

    return total.mean()