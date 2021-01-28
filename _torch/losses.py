import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
# from itertools import ifilterfalse
from pytorch_toolbelt.losses import LovaszLoss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    
    def forward(self, inputs, target):
        N = target.size(0)
        smooth = 1
        input_flat = inputs.view(N, -1)
        target_flat = target.view(N, -1)
        
        intersection = input_flat * target_flat
        
        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N
        
        return loss


class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """
    
    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()
    
    def forward(self, inputs, target, weights=None):
        
        target = torch.nn.functional.one_hot(target.long()).permute(0, 3, 1, 2)
        inputs = torch.nn.functional.softmax(inputs, dim=1)
        
        C = target.shape[1]
        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes
        
        dice = DiceLoss()
        totalLoss = 0
        
        for i in range(C):
            diceLoss = dice(inputs[:, i], target[:, i])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss
        
        return totalLoss


class Iou_pytorch(torch.nn.Module):
    """ Calculate SNR between target and the output image."""
    
    def __init__(self):
        super(Iou_pytorch, self).__init__()
    
    def forward(self, output, target):
        output = torch.sigmoid(output).round().int()
        target = target.round().int()
        smooth = 1e-6
        intersection = (output & target).float().sum((1, 2, 3, 4))
        union = (output | target).float().sum((1, 2, 3, 4))
        iou = (intersection + smooth) / (union + smooth)
        #         thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10
        
        return iou.mean()


class lovasz_hinge(torch.nn.Module):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    
    def __init__(self, per_image=True, ignore=None):
        super(lovasz_hinge, self).__init__()
        self.per_image = per_image
        self.ignore = ignore
    
    def forward(self, logits, labels):
        
        if self.per_image:
            loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), self.ignore))
                        for log, lab in zip(logits, labels))
        else:
            loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, self.ignore))
        return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszBCELoss(torch.nn.Module):
    def __init__(self, lovasz_weight=0.75, ce_weight=0.25):
        super().__init__()
        self.lovasz_weight = lovasz_weight
        self.ce_weight = ce_weight
        self.ce = torch.nn.CrossEntropyLoss()
        self.lovasz = LovaszLoss()

    def forward(self, output, target):
        if self.lovasz_weight > 0:
            lovasz = self.lovasz(torch.softmax(output, dim=1), target) * self.lovasz_weight
        else:
            lovasz = 0

        if self.ce_weight > 0:
            ce = self.ce(output, target.long()) * self.ce_weight
        else:
            ce = 0

        return lovasz + ce


def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


class F1Loss(torch.nn.Module):
    def __init__(self, weights:list=None, eps:float=1e-10):
        self.weights = weights
        self.eps = eps
    
    def forward(self, output, target):
    
    
