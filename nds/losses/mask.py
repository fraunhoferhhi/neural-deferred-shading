import torch
from typing import Dict, List

from nds.core import View

def mask_loss(views: List[View], gbuffers: List[Dict[str, torch.Tensor]], loss_function = torch.nn.MSELoss()):
    """ Compute the mask term as the mean difference between the original masks and the rendered masks.
    
    Args:
        views (List[View]): Views with masks
        gbuffers (List[Dict[str, torch.Tensor]]): G-buffers for each view with the 'mask' channel
        loss_function (Callable): Function for comparing the masks or generally a set of pixels
    """

    loss = 0.0
    for view, gbuffer in zip(views, gbuffers):
        loss += loss_function(view.mask, gbuffer["mask"])
    return loss / len(views)