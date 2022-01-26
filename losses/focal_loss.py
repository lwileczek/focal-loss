import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def reweight(cls_num_list, beta=0.9999):
    """
    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return:
    """
    per_cls_weights = None

    c = len(cls_num_list)
    per_cls_weights = np.zeros(c)
    for i,n in enumerate(cls_num_list):
        per_cls_weights[i] =  (1 - beta) / (1 - beta**n)  # a is inverse of En
    # Normalize so sum_1^C \alpha = C
    per_cls_weights *= (c / np.sum(per_cls_weights))

    return torch.from_numpy(per_cls_weights)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        """
        loss = None

        """
        The original Focal Loss Paper provides code for how to use Focal Loss:
          - https://www.paperswithcode.com/method/focal-loss
          - https://github.com/clcarwin/focal_loss_pytorch/blob/e11e75bad957aecf641db6998a1016204722c1bb/focalloss.py#L6

        I used this to help me write the class-balanced CE Loss, However I try to write
        comments to show my understanding of the code.
        """
        N, _ = input.shape
        log_probs = F.log_softmax(input)
        log_probs = log_probs[np.arange(N), target]
        probs = log_probs.data.exp()
        loss = -1 * self.weight[target]*(1-probs)**self.gamma * log_probs

        return loss.sum()
