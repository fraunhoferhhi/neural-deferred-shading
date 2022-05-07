import torch
import torch.nn as nn
from typing import Callable, Dict, List

from nds.core import View

class RandomSamples(object):
    def __init__(self, h, w, percentage=.5):
        self.idx = torch.randperm(h*w)[:int(h*w*percentage)]

    def __call__(self, tensor):
        """ Select samples from a tensor.

        Args:
            tensor: Tensor to select samples from (HxWxC or NxC)
        """
        return tensor.view(-1, tensor.shape[-1])[self.idx]

def shading_loss(views: List[View], gbuffers: List[Dict[str, torch.Tensor]], shader, loss_function=torch.nn.L1Loss(), shading_percentage=1):
    """ Compute the shading term as the mean difference between the original images and the rendered images from a shader.
    
    Args:
        views (List[View]): Views with color images and masks
        gbuffers (List[Dict[str, torch.Tensor]]): G-buffers for each view with 'mask', 'position', and 'normal' channels
        shader (Callable): Shader that generates colors from the G-buffer data
        loss_function (Callable): Function for comparing the images or generally a set of pixels
        shading_percentage (float): Percentage of (random) valid pixels that are shaded; 
                                    pixels are valid if they are contained in the original and G-buffer masks (range is 0-1).
    """

    loss = 0
    sample_fn = lambda x: x
    for view, gbuffer in zip(views, gbuffers):
        # Get valid area
        mask = ((view.mask > 0) & (gbuffer["mask"] > 0)).squeeze()

        # Sample only within valid area
        if shading_percentage != 1:
            sample_fn = RandomSamples(view.mask[mask].shape[0], 1, shading_percentage)
    
        target = sample_fn(view.color[mask])

        position = sample_fn(gbuffer["position"][mask])
        normal = sample_fn(gbuffer["normal"][mask])

        view_direction = view.camera.center - position
        view_direction = torch.nn.functional.normalize(view_direction, dim=-1)

        loss += loss_function(shader(position, normal, view_direction), target)

    return loss / len(views)
        