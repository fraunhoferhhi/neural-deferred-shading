import math
import numpy as np

from .camera import PerspectiveCamera
from .utils import ray_plane_intersection

class OrbitControl:
    def __init__(self, camera: PerspectiveCamera):
        self.camera = camera
    
        # Extract the view direction and from that distance, elevation and azimuth
        v = self.camera.eye - self.camera.lookat
        self.distance = np.linalg.norm(v) # override

        v = v / self.distance
        self.elevation = math.acos(v[1])

        v[1] = 0.0
        v = v / np.linalg.norm(v)
        self.azimuth = math.atan2(-v[2], v[0])

    def update_camera(self):
        self.camera.eye = self.camera.lookat + np.array([
            self.distance * math.cos(self.azimuth) * math.sin(self.elevation),
            self.distance * math.cos(self.elevation),
            -self.distance * math.sin(self.azimuth) * math.sin(self.elevation),
        ])
        self.camera.mark_dirty()

    def handle_drag(self, from_x, to_x, from_y, to_y, button):
        dx = to_x - from_x
        dy = to_y - from_y

        if button == 0: # LMB
            self.azimuth = self.azimuth - dx * 0.01;

            if self.azimuth < 0.0:
                self.azimuth = 2*np.pi + self.azimuth
            elif self.azimuth >= 2*np.pi:
                self.azimuth = 0.0

            self.elevation = max(0.001, min(self.elevation - dy * 0.01, np.pi))

            self.update_camera()
        elif button == 1: # RMB
            P_inv = np.linalg.inv(np.array(self.camera.projection_matrix).T)
            V_inv = np.linalg.inv(np.array(self.camera.view_matrix).T)

            def unproject(x, y):
                x_ndc = 2 * x / (self.camera.viewport[2] - self.camera.viewport[0]) - 1
                y_ndc = -(2 * y / (self.camera.viewport[3] - self.camera.viewport[1]) - 1)
                l = (V_inv @ P_inv @ np.array([x_ndc, y_ndc, 1.0, 1.0]))[:3]
                return l / np.linalg.norm(l)

            # # xz-plane at lookat
            # p0 = self.camera.lookat
            # n = np.array([0.0, 1.0, 0.0])

            # Plane perpendicular to the image at `lookat`
            p0 = self.camera.lookat
            n = V_inv[:3, :3] @ np.array([[0.0], [0.0], [1.0]])

            # 
            from_ray = (self.camera.eye, unproject(from_x, from_y))
            to_ray = (self.camera.eye, unproject(to_x, to_y))

            from_p = from_ray[0] + from_ray[1]*ray_plane_intersection(p0, n, *from_ray)
            to_p = to_ray[0] + to_ray[1]*ray_plane_intersection(p0, n, *to_ray)

            self.camera.lookat = self.camera.lookat + (from_p - to_p)

            self.update_camera()

    def handle_scroll(self, dx, dy):
        # We interpret the distance as function of mouse wheel configuration f(w).
        # A change of the wheel position corresponds to a step into the gradient direction df/dw.
        # The gradient should be proportional to the function itself df/dw := a*f(w),
        # so small steps for small distances, large steps for large distances.
        # (Side note: the function is the exponential f(w) = a*exp(w) + b.)
        self.distance = max(0.01, self.distance + 0.3 * self.distance * -dy) # 0.6

        self.update_camera()