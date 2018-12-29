import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
__all__ = ['WeightSmoothL1Loss', 'SoftmaxFocalLoss', 'FocalLoss', 'MultiTaskLoss']


class WeightSmoothL1Loss(nn.SmoothL1Loss):
    r'''Compute Smooth L1 Loss for offset parameters regression
    modified by cxg
    inputs {predict tensor} : format = [batch_size, channel, height, width]
    targets {label tensor} : format = [batch_size, channel, height, width]
    w_in { input weight tensor} : format = [batch_size, channel, height, width] , value is 0 or 1
    w_in { output weight tensor} : format = [batch_size, channel, height, width] , value is 0 or 1

    reduce {bool} : true for training with multi gpu , false for one gpu
        if false , then the return is a [batch_size, 1] tensor
    size_average {bool} : if true , loss is averaged over all non-ignored pixels
                          if false , loss is averaged over batch size
    '''
    def __init__(self, size_average=True, reduce=True):
        super(WeightSmoothL1Loss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, inputs, targets, w_in, w_out, sigma=1.):  # sigma=10 or 1?
        # print inputs.size()
        targets.requires_grad = False
        w_in.requires_grad = False
        w_out.requires_grad = False
        sigma2 = sigma*sigma
        diff = (inputs - targets)*w_in
        abs_diff = diff.abs()
        output = abs_diff.lt(1.0/sigma2).float()*torch.pow(diff, 2) * \
            0.5*sigma2 + abs_diff.ge(1.0/sigma2).float()*(abs_diff-0.5/sigma2)
        output = output*w_out
        if self.reduce:
            if self.size_average:
                return output.sum()/w_in.ge(0).float().sum()
            else:
                return output.sum()/inputs.size(0)
        else:
            return output.view(output.size(0), -1).sum(1).unsqueeze(1)


class SoftmaxFocalLoss(nn.Module):
    r'''Compute multi-label loss using softmax_cross_entropy_with_logits
    focal loss for Multi-Label classification
    FL(p_t) = -alpha * (1-p_t)^{gamma} * log(p_t)
    :param
    predict {predict tensor}: format=[batch_size, channel, height, width], channel is number of class
    targeet {lebel tensor}: format=[batch_size, channel, height, width], channel is number of class, one-hot tensor
    reduce {bool} : true for training with multi gpu , false for one gpu
        if false , then the return is a [batch_size, 1] tensor
    size_average {bool} : if true , loss is averaged over all non-ignored pixels
                          if false , loss is averaged over batch size
    '''
    def __init__(self, alpha=0.5, gamma=3.0, size_average=True, reduce=False):
        super(SoftmaxFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, predict, target):
        target.requires_grad = False

        mask = target.ge(0)
        onehot = target.float()*mask.float()

        logits = nn.functional.softmax(predict, dim=1) + 1.e-9
        log_data = onehot*logits.log()
        pow_data = onehot*torch.pow(1-logits, self.gamma)
        loss = -self.alpha*pow_data*log_data*mask.float()

        if self.reduce:
            if self.size_average:
                return loss.sum()/mask.float().sum()
            else:
                return loss.sum()/mask.size(0)
        else:
            return loss.view(loss.size(0), -1).sum(1).unsqueeze(1)


class FocalLoss(nn.Module):
    r"""
    This criterion is a implementaion of Focal Loss, which is proposed in Focall Loss for Dense Object Detection.

        Loss(x, class) = - \alpha*(1-softmax(x)[class])^\gamma*\log(softmax(x)[class])

    The losses are averaged across observations for each minibatch

    Args:
        alpha: scalar, the scalar factor for this criterion
        gamma: float/double, reduces the relative loss for well-classified examples (p > 0.5),
            putting more focus on hard, misclassified examples
        size_average: bool, by default, the losses are averaged over observations for each minibatch,
            if size_average is False, the losses are summed for each minibatch
    """

    def __init__(self, alpha=0.800000011921, gamma=3.0, size_average=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, inputs, targets):

        p = inputs.sigmoid()
        # print p.size(), p
        term1 = torch.pow(1-p, self.gamma)*(p.log())
        term2 = torch.pow(p, self.gamma)*(-inputs*inputs.ge(0).float() -
                                          (1+(inputs-2*inputs*inputs.ge(0).float()).exp()).log())
        loss = - self.alpha * \
            targets.eq(1).float()*term1 - (1-self.alpha) * \
            targets.eq(0).float()*term2

        if self.reduce:
            if self.size_average:
                return loss.sum()/targets.ge(0).float().sum()
            else:
                return loss.sum()/targets.size(0)
        else:
            return loss.view(loss.size(0), -1).sum(1).unsqueeze(1)


class MultiTaskLoss(nn.Module):
    def __init__(self, num_losses):
        super(MultiTaskLoss, self).__init__()
        self.sigma = Parameter(torch.ones(num_losses)/num_losses)

    def forward(self, losses):
        final_loss = 0
        for i, loss in enumerate(losses):
            final_loss += 1./(2*self.sigma[i].pow(2)) * \
                loss + 0.5*(self.sigma[i].pow(2)+1).log()
        return final_loss


class CustomLoss(nn.Module):
    def __init__(self, device='cpu', num_classes=1):
        super(CustomLoss, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.loss_cls_fn = SoftmaxFocalLoss()
        self.loss_loc_fn = torch.nn.SmoothL1Loss(reduce=True, size_average=False)

    def forward(self, preds, targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).
        Args:
          preds: (tensor)  cls_preds + reg_preds, sized[batch_size, height, width, 7]
          cls_preds: (tensor) predicted class confidences, sized [batch_size, height, width, 1].
          cls_targets: (tensor) encoded target labels, sized [batch_size, height, width, 1].
          loc_preds: (tensor) predicted target locations, sized [batch_size, height, width, 6 or 8].
          loc_targets: (tensor) encoded target locations, sized [batch_size, height, width, 6 or 8].
        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        '''
        cls_preds = preds[..., 0]
        loc_preds = preds[..., 1:]

        cls_targets = targets[..., 0]
        loc_targets = targets[..., 1:]

        cls_loss = self.loss_cls_fn(cls_preds, cls_targets)
        loc_loss = self.loss_loc_fn(loc_preds, loc_targets)

        return cls_loss+loc_loss, cls_loss, loc_loss

def test_loss():
    loss = CustomLoss()
    pred = torch.sigmoid(torch.randn(1, 800, 700, 7))
    # label = torch.tensor([[[[1, 0.4, 0.5], [0, 0.2, 0.5]], [[0, 0.1, 0.1], [1, 0.8, 0.4]]]])
    label = torch.randn(1, 800, 700, 7)
    loss = loss(pred, label)
    print("loss ... ", loss)

def loss_fun():
    loss_fn = torch.nn.SmoothL1Loss(reduce=True, size_average=True)
    inputs = torch.autograd.Variable(torch.randn(1,800, 700, 6))
    target = torch.autograd.Variable(torch.randn(1,800, 700, 6))
    loss = loss_fn(inputs, target)
    print(loss)
    print(inputs.size(), target.size(), loss.size())


if __name__ == '__main__':
    test_loss()