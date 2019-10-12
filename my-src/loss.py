import numpy as np
from torch.nn import functional as F
from torch import nn


def hard_negative_mining(pred, target, weight=None):
    """
    Online hard mining on the entire batch

    :param pred: predicted character or affinity heat map, torch.cuda.FloatTensor, shape = [num_pixels]
    :param target: target character or affinity heat map, torch.cuda.FloatTensor, shape = [num_pixels]
    :param weight: If weight is not None, it denotes the weight given to each pixel for weak-supervision training
    :return: Online Hard Negative Mining loss
    """
    cpu_target = target.data.cpu().numpy()
    all_loss = F.mse_loss(pred, target, reduction="none")

    positive = np.where(cpu_target >= config.THRESHOLD_POSITIVE)[0]
    negative = np.where(cpu_target <= config.THRESHOLD_NEGATIVE)[0]

    positive_loss = all_loss[positive]
    negative_loss = all_loss[negative]

    negative_loss_cpu = np.argsort(-negative_loss.data.cpu().numpy())[
        0 : min(max(1000, 3 * positive_loss.shape[0]), negative_loss.shape[0])
    ]

    return (positive_loss.sum() + negative_loss[negative_loss_cpu].sum()) / (
        positive_loss.shape[0] + negative_loss_cpu.shape[0]
    )


class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()

    def forward(self, y, gt):
        """
        Args:
            y: (N, 1, H /2 , W /2)
            gt: (N, 1, H /2 , W /2)
        """
        N, C, H, W = y.shape

        # output = (
        #     output.permute(0, 2, 3, 1)
        #     .contiguous()
        #     .view([batch_size * height * width, channels])
        # )

        # character = output[:, 0]
        # affinity = output[:, 1]

        # affinity_map = affinity_map.view([batch_size * height * width])
        # character_map = character_map.view([batch_size * height * width])

        # loss = hard_negative_mining(y, gt)

        return loss
