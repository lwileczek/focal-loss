import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def reweight(cls_num_list, beta=0.9999):
    '''
    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return:
    '''
    per_cls_weights = None
    #############################################################################
    #       reweight each class by effective numbers                            #
    #############################################################################
    c = len(cls_num_list)
    per_cls_weights = np.zeros(c)
    for i,n in enumerate(cls_num_list):
        per_cls_weights[i] =  (1 - beta) / (1 - beta**n)  # a is inverse of En
    # Normalize so sum_1^C \alpha = C
    per_cls_weights *= (c / np.sum(per_cls_weights))
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return torch.from_numpy(per_cls_weights)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        '''
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        '''
        loss = None
        #############################################################################
        #       Implement forward pass of the focal loss                            #
        #############################################################################
        """
        The original Focal Loss Paper provides code for how to use Focal Loss:
          - https://www.paperswithcode.com/method/focal-loss
          - https://github.com/clcarwin/focal_loss_pytorch/blob/e11e75bad957aecf641db6998a1016204722c1bb/focalloss.py#L6

        I used this to help me write the class-balanced CE Loss, However I try to write
        comments to show my understanding of the code.
        """
        N, _ = input.shape
        # print("Shape of inputs:", input.shape)
        # print("Shape of targets:", target.shape)
        log_probs = F.log_softmax(input)    # Take the softmax and log of model outputs
        # https://pytorch.org/docs/stable/generated/torch.gather.html
        log_probs = log_probs[np.arange(N), target]  # Select from our labels the our models prediction of the true value
        probs = log_probs.data.exp()  # Exponentiate to get the probababilities for each class
        # Focal Loss with the weights being pulled from our class-balanced formula
        loss = -1 * self.weight[target]*(1-probs)**self.gamma * log_probs
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss.sum()
