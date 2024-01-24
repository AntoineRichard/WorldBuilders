import numpy as np
import quaternion
from .Types import *

# I was in troublem deciding whether I should inherit BaseSampler or create new class and inherit it. 
# Since Z-value clip function is deterministic, meaning z value is always determined based on heightmap. 
# There is no point in creating random distribution as done in BaseSampler and other inheriting class. 
# Thus, I made BaseClipper method and new class called HeightClipper and NormalMapClippter which inherit BaseClipper. 

class BaseClipper:
    def __init__(self, sampler_cfg: Sampler_T):
        self._sampler_cfg = sampler_cfg
        self.image = self._sampler_cfg.data
        self.resolution = self._sampler_cfg.resolution
        self.mpp_resolution = self._sampler_cfg.mpp_resolution

        assert len(self.image.shape) == 2, f"image need to be 1 channel image, not {self.image.shape}"
    
    def __call__(self, **kwargs):
        return self.sample(**kwargs)

    def sample(self, **kwargs):
        raise NotImplementedError()

class HeightClipper(BaseClipper):
    def __init__(self, sampler_cfg: Sampler_T):
        super().__init__(sampler_cfg)

    def sample(self, query_point:np.ndarray, **kwargs):
        """
        query point is (x, y) point generated from 2D sampler. 
        """
        x = query_point[:, 0]
        y = query_point[:, 1]
        H, W = self.resolution
        # from xy to uv coordinate
        us = x // self.mpp_resolution #horizontal
        vs = H * np.ones_like(y) - y // self.mpp_resolution #vertical
        images = []
        for u, v in zip(us, vs):
            u = int(u)
            v = int(v)
            images.append(self.image[v, u])
        return np.stack(images)[:, np.newaxis]

class NormalMapClipper(BaseClipper):
    def __init__(self, sampler_cfg: Sampler_T):
        super().__init__(sampler_cfg)

    def compute_slopes(self)->None:
        nx,ny = np.gradient(self.image)
        slope_x = np.arctan2(nx,1) #theta_x = tan^-1(nx)
        slope_y = np.arctan2(ny,1) #theta_y = tan^-1(ny)
        # magnitude = np.hypot(nx,ny)
        # slope_xy = np.arctan2(magnitude,1)
        self.slope_x = np.rad2deg(slope_x)
        self.slope_y = np.rad2deg(slope_y)
        # self.slope_xy = np.rad2deg(slope_xy)
        # self.magnitude = magnitude

    def sample(self, query_point:np.ndarray, **kwargs):
        """
        query point is (x, y) point generated from sampler
        """
        self.compute_slopes()
        x = query_point[:, 0]
        y = query_point[:, 1]
        H, W = self.resolution
        us = x // self.mpp_resolution #horizontal
        vs = H * np.ones_like(y) - y // self.mpp_resolution #vertical
        quat = []
        for u, v in zip(us, vs):
            u = int(u)
            v = int(v)
            roll = self.slope_y[v, u]
            pitch = self.slope_x[v, u]
            yaw = 0.0
            q = quaternion.as_float_array(quaternion.from_euler_angles([roll, pitch, yaw]))
            quat.append(q)

        return np.stack(quat)


class ClipMapClipper(BaseClipper):
    def __init__(self, sampler_cfg: Sampler_T):
        super().__init__(sampler_cfg)
        self.image = np.flipud(self.image)
    
    @staticmethod
    def _linear_interpolation(
        dx: np.ndarray,
        dy: np.ndarray,
        q11: np.ndarray,
        q12: np.ndarray,
        q21: np.ndarray,
        q22: np.ndarray,
    ):
        return (1-dy)*((1-dx)*q11+dx*q21)+dy*((1-dx)*q12+dx*q22)
  
    def sample(self, query_point:np.ndarray, **kwargs):
        """
        query point is (x, y) point generated from 2D sampler. 
        query point is relative to bottom left corner.
        Note that in clipmap, world origin is not aligned with pixel origin.
        Therefore, need to add offset to query_point
        """
        points = []
        qx = query_point[:, 0]
        qy = query_point[:, 1]
        for point_x, point_y in zip(qx, qy):
            x = point_x/self.mpp_resolution + self.resolution[1]//2
            y = point_y/self.mpp_resolution + self.resolution[0]//2

            # sampled discrete points should not be bigger than image boundary
            # which is 0<=x<=W-1, 0<=y<=H-1
            x = np.minimum(x, self.resolution[1] - 1)
            y = np.minimum(y, self.resolution[0] - 1)
            x = np.maximum(x, 0)
            y = np.maximum(y, 0)

            ########################
            # (x1, y2) ---- (x2, y1)
            #   |              |
            #   |     (x, y)   |
            #   |              |
            # (x1, y2) ---- (x2, y2)
            ########################
            x1 = np.trunc(x).astype(int)
            y1 = np.trunc(y).astype(int)
            x2 = np.minimum(x1 + 1, self.resolution[1] - 1)
            y2 = np.minimum(y1 + 1, self.resolution[0] - 1)
            dx = x - x1
            dy = y - y1

            # q11 = self.image[x1, y1]
            # q12 = self.image[x1, y2]
            # q21 = self.image[x2, y1]
            # q22 = self.image[x2, y2]

            q11 = self.image[y1, x1]
            q12 = self.image[y2, x1]
            q21 = self.image[y1, x2]
            q22 = self.image[y2, x2]

            z = self._linear_interpolation(dx, dy, q11, q12, q21, q22)
            points.append(z)
        return np.stack(points)[:, np.newaxis]


class ClipperFactory:
    def __init__(self):
        self.creators = {}
    
    def register(self, name: str, class_: BaseClipper) -> None:
        self.creators[name] = class_
        
    def get(self, cfg: Sampler_T, **kwargs:dict) -> BaseClipper:
        if cfg.__class__.__name__ not in self.creators.keys():
            raise ValueError("Unknown sampler requested.")
        return self.creators[cfg.__class__.__name__](cfg)

Clipper_Factory = ClipperFactory()
Clipper_Factory.register("ImageClipper_T", HeightClipper)
Clipper_Factory.register("NormalMapClipper_T", NormalMapClipper)
Clipper_Factory.register("ClipMapClipper_T", ClipMapClipper)