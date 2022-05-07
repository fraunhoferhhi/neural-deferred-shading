import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torch

from nds.core import Camera

class View:
    """ A View is a combination of camera and image(s).

    Args:
        color (tensor): RGB color image (WxHx3)
        mask (tensor): Object mask (WxHx1)
        camera (Camera): Camera associated with this view
        device (torch.device): Device where the images and camera are stored
    """

    def __init__(self, color, mask, camera, device='cpu'):
        self.color = color.to(device)
        self.mask = mask.to(device)
        self.camera = camera.to(device)
        self.device = device

    @classmethod
    def load(cls, image_path, device='cpu'):
        """ Load a view from a given image path.

        The paths of the camera matrices are deduced from the image path. 
        Given an image path `path/to/directory/foo.png`, the paths to the camera matrices
        in numpy readable text format are assumed to be `path/to/directory/foo_k.txt`, 
        `path/to/directory/foo_r.txt`, and `path/to/directory/foo_t.txt`.

        Args:
            image_path (Union[Path, str]): Path to the image file that contains the color and optionally the mask
            device (torch.device): Device where the images and camera are stored
        """

        image_path = Path(image_path)

        # Load the camera
        K = np.loadtxt(image_path.parent / (image_path.stem + "_k.txt"))
        R = np.loadtxt(image_path.parent / (image_path.stem + "_r.txt"))
        t = np.loadtxt(image_path.parent / (image_path.stem + "_t.txt"))
        camera = Camera(K, R, t)
        
        # Load the color
        color = torch.FloatTensor(np.array(Image.open(image_path)))
        color /= 255.0
        
        # Extract the mask
        if color.shape[2] == 4:
            mask = color[:, :, 3:]
        else:
            mask = torch.ones_like(color[:, :, 0:1])

        color = color[:, :, :3]

        return cls(color, mask, camera, device=device)

    def to(self, device: str = "cpu"):
        self.color = self.color.to(device)
        self.mask = self.mask.to(device)
        self.camera = self.camera.to(device)
        self.device = device
        return self

    @property
    def resolution(self):
        return (self.color.shape[0], self.color.shape[1])
    
    def scale(self, inverse_factor):
        """ Scale the view by a factor.
        
        This operation is NOT differentiable in the current state as 
        we are using opencv.

        Args:
            inverse_factor (float): Inverse of the scale factor (e.g. to halve the image size, pass `2`)
        """
        
        scaled_height = self.color.shape[0] // inverse_factor
        scaled_width = self.color.shape[1] // inverse_factor

        scale_x = scaled_width / self.color.shape[1]
        scale_y = scaled_height / self.color.shape[0]
        
        self.color = torch.FloatTensor(cv2.resize(self.color.cpu().numpy(), dsize=(scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)).to(self.device)
        self.mask = torch.FloatTensor(cv2.resize(self.mask.cpu().numpy(), dsize=(scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST)).to(self.device)
        self.mask = self.mask.unsqueeze(-1) # Make sure the mask is HxWx1

        self.camera.K = torch.FloatTensor(np.diag([scale_x, scale_y, 1])).to(self.device) @ self.camera.K  
    
    def transform(self, A, A_inv=None):
        """ Transform the view pose with an affine mapping.

        Args:
            A (tensor): Affine matrix (4x4)
            A_inv (tensor, optional): Inverse of the affine matrix A (4x4)
        """

        if not torch.is_tensor(A):
            A = torch.from_numpy(A)
        
        if A_inv is not None and not torch.is_tensor(A_inv):
            A_inv = torch.from_numpy(A_inv)

        A = A.to(self.device, dtype=torch.float32)
        if A_inv is not None:
            A_inv = A_inv.to(self.device, dtype=torch.float32)

        if A_inv is None:
            A_inv = torch.inverse(A)

        # Transform camera extrinsics according to  [R'|t'] = [R|t] * A_inv.
        # We compose the projection matrix and decompose it again, to correctly
        # propagate scale and shear related factors to the K matrix, 
        # and thus make sure that R is a rotation matrix.
        R = self.camera.R @ A_inv[:3, :3]
        t = self.camera.R @ A_inv[:3, 3] + self.camera.t
        P = torch.zeros((3, 4), device=self.device)
        P[:3, :3] = self.camera.K @ R
        P[:3, 3] = self.camera.K @ t
        K, R, c, _, _, _, _ = cv2.decomposeProjectionMatrix(P.cpu().detach().numpy())
        c = c[:3, 0] / c[3]
        t = - R @ c

        # ensure unique scaling of K matrix
        K = K / K[2,2]
        
        self.camera.K = torch.from_numpy(K).to(self.device)
        self.camera.R = torch.from_numpy(R).to(self.device)
        self.camera.t = torch.from_numpy(t).to(self.device)
        
    def project(self, points, depth_as_distance=False):
        """ Project points to the view's image plane according to the equation x = K*(R*X + t).

        Args:
            points (torch.tensor): 3D Points (A x ... x Z x 3)
            depth_as_distance (bool): Whether the depths in the result are the euclidean distances to the camera center
                                      or the Z coordinates of the points in camera space.
        
        Returns:
            pixels (torch.tensor): Pixel coordinates of the input points in the image space and 
                                   the points' depth relative to the view (A x ... x Z x 3).
        """

        # 
        points_c = points @ torch.transpose(self.camera.R, 0, 1) + self.camera.t
        pixels = points_c @ torch.transpose(self.camera.K, 0, 1)
        pixels = pixels[..., :2] / pixels[..., 2:]
        depths = points_c[..., 2:] if not depth_as_distance else torch.norm(points_c, p=2, dim=-1, keepdim=True)
        return torch.cat([pixels, depths], dim=-1)