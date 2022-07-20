import numpy as np
from pyrr import Matrix44

class PerspectiveCamera:
    def __init__(self, viewport):
        self.eye = np.array([0, 0, 2])
        self.lookat = np.array([0, 0, 0])
        self.up = np.array([0, 1, 0])

        self.fov = 30
        self.near = 0.1
        self.far = 1000

        self._viewport = viewport

        self.mark_dirty()

    def mark_dirty(self):
        self.mark_view_dirty()
        self.mark_projection_dirty()

    def mark_view_dirty(self):
        self._view_matrix = None

    def mark_projection_dirty(self):
        self.projection_matrix_ = None

    @property
    def viewport(self):
        return self._viewport

    @viewport.setter
    def viewport(self, v):
        self._viewport = v
        self.mark_dirty()

    @property
    def view_matrix(self):
        if self._view_matrix is None:
            self._view_matrix = Matrix44.look_at(self.eye, self.lookat, self.up)            

        return self._view_matrix

    @property
    def projection_matrix(self):
        if self.projection_matrix_ is None:
            width = self._viewport[2] - self._viewport[0]
            height = self._viewport[3] - self._viewport[1]
            aspect_ratio = width / height
            self.projection_matrix_ = Matrix44.perspective_projection(self.fov, aspect_ratio, self.near, self.far)

        return self.projection_matrix_