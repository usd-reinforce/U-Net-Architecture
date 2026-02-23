import torch
import torch.nn.functional as F

def dice_loss(pred, target, smooth: int = 1):
    """
    computes the dice loss for binary segmentation.
    :param pred: Tensor fo predictions
    :param target:
    :param smooth:
    :return:
    """

    pass