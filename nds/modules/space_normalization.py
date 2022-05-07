import numpy as np
import torch
from typing import List

from nds.core import Mesh, View
from nds.utils.geometry import AABB, normalize_aabb

class SpaceNormalization:
    def __init__(self, points_3d):
        self.aabb = AABB(points_3d.cpu().numpy() if torch.is_tensor(points_3d) else points_3d)
        self.A, self.A_inv = normalize_aabb(self.aabb, side_length=2)
    
    def normalize_views(self, views: List[View]):
        for view in views:
            view.transform(self.A, self.A_inv)
        return views
    
    def normalize_mesh(self, mesh: Mesh):
        v_normalized = mesh.vertices.cpu().numpy() @ self.A[:3, :3].T + self.A[:3, 3][np.newaxis, :]
        return mesh.with_vertices(v_normalized)

    def normalize_aabb(self, aabb: AABB):
        return AABB(aabb.corners @ self.A[:3, :3].T + self.A[:3, 3][np.newaxis, :])

    def denormalize_mesh(self, mesh: Mesh):
        v_denormalized = mesh.vertices.cpu().numpy() @ self.A_inv[:3, :3].T + self.A_inv[:3, 3][np.newaxis, :]
        return mesh.with_vertices(v_denormalized)