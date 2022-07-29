import torch
import torch.nn as nn
import numpy as np

class TextureLoss(nn.Module):
    """
    Loss function that applies a weighted Cross Entropy Loss to
    the outermost layer of the voxel model 
    """

    def __init__(self, pos_weight: int = 10) -> None:
        """
        :param: pos_weight: Weight to apply to underepresented class label
        """

        super(TextureLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(weight=torch.Tensor([1, pos_weight]), ignore_index=2, reduction='none')

    def forward(self, preds: torch.Tensor, targs: torch.Tensor) -> float:

        loss = self.loss(preds, targs)
        loss = torch.mean(loss, 1)

        return loss

def classification_error(preds: torch.Tensor, targs: torch.Tensor) -> torch.Tensor:
    """
    Provides the classification error for a prediction: i.e. the number of 
    incorrect predictions over the total number of predictions
    """

    _, pred_class = torch.max(preds, dim=1)
    errors = []
    for pred, targ in zip(pred_class, targs):
        mask = targ != 2
        masked_pred = pred.masked_select(mask)
        masked_targ = targ.masked_select(mask)
        if torch.sum(masked_pred) == 0:  # ignore degenerate masks
            error = torch.tensor(1000).to(device=preds.device, dtype=preds.dtype)
        else:
            masked_targ = masked_targ.type(torch.int64)
            error = masked_pred != masked_targ
            error = torch.mean(error.to(dtype=preds.dtype))
        errors.append(error)
    errors = torch.stack(errors)
    return errors

class DiverseLoss(nn.Module):
    def __init__(self, pos_weight: int = 10, beta=1, is_train=True, eval_mode=False):
        """
        :param pos_weight:
        :param: beta: 
        :param is_train
        :param eval_mode: returns match indices, and computes loss as L2 loss
        """
        super(DiverseLoss, self).__init__()
        self.loss = classification_error if eval_mode \
            else TextureLoss(pos_weight=pos_weight)
        self.beta = beta
        self.is_train = is_train
        if eval_mode:
            self.is_train = False

    def forward(self, preds: torch.Tensor, targs: torch.Tensor):
        """
        :param preds: N x Ep x 2 x P (torch.float32)
        :param targs: N x Et x P (torch.int64)
        :return: loss, match_indices
        """

        preds = preds.view(*preds.shape[:3], -1)
        targs = targs.view(*targs.shape[:2], -1)
        N, Ep, _, P = preds.shape
        _, Et, _ = targs.shape
        
        losses = []
        match_indices = []
        for pred, targ in zip(preds, targs):
            pred = pred.repeat(Et, 1, 1)
            targ = targ.repeat(1, Ep).view(-1, P)
            loss_matrix = self.loss(pred, targ).view(Et, Ep)
            loss, match_idx = loss_matrix.min(1)
            loss = loss.mean()
            if self.is_train:
                l_catchup = loss_matrix.min(0)[0].max() / Ep
                loss = loss + self.beta * l_catchup
            losses.append(loss)
            match_indices.append(match_idx)
        loss = torch.stack(losses).mean()
        match_indices = torch.stack(match_indices)
        
        return loss, match_indices

