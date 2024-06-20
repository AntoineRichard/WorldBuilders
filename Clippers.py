import numpy as np
import quaternion
from .Types import *

class BaseClipper:
    def __init__(self, clipper_cfg: Clipper_T):
        self._clipper_cfg = clipper_cfg
        self.image = self._clipper_cfg.data
        self.resolution = self._clipper_cfg.resolution
        self.mpp_resolution = self._clipper_cfg.mpp_resolution

        assert len(self.image.shape) == 2, f"image need to be 1 channel image, not {self.image.shape}"
    
    def __call__(self, **kwargs):
        return self.sample(**kwargs)

    def sample(self, **kwargs):
        raise NotImplementedError()

class HeightClipper(BaseClipper):
    def __init__(self, clipper_cfg: Clipper_T):
        super().__init__(clipper_cfg)

    def sample(self, query_point:np.ndarray, **kwargs):
        """
        query point is (x, y) point generated from 2D sampler. 
        """
        height = []
        x = query_point[:, 0]
        y = query_point[:, 1]
        H, W = self.resolution
        # from xy (cartesian) to uv coordinate
        if self._clipper_cfg.loc_origin == "lower":
            us = x // self.mpp_resolution #horizontal
            vs = (H - 1) * np.ones_like(y) - y // self.mpp_resolution #vertical
        elif self._clipper_cfg.loc_origin == "upper":
            us = x // self.mpp_resolution
            vs = y // self.mpp_resolution
        elif self._clipper_cfg.loc_origin == "center":
            us = W //2 + x // self.mpp_resolution
            vs = H//2 + y // self.mpp_resolution
        for u, v in zip(us, vs):
            u = int(u)
            v = int(v)
            height.append(self.image[v, u])
        return np.stack(height)[:, np.newaxis]

class NormalMapClipper(BaseClipper):
    def __init__(self, clipper_cfg: Clipper_T):
        super().__init__(clipper_cfg)
        self.compute_slopes()

    def compute_slopes(self)->None:
        nx,ny = np.gradient(self.image)
        slope_x = np.arctan2(nx,1) #theta_x = tan^-1(nx)
        slope_y = np.arctan2(ny,1) #theta_y = tan^-1(ny)
        self.slope_x = np.rad2deg(slope_x)
        self.slope_y = np.rad2deg(slope_y)

    def sample(self, query_point:np.ndarray, **kwargs):
        """
        query point is (x, y) point generated from sampler
        """
        quat = []
        x = query_point[:, 0]
        y = query_point[:, 1]
        H, W = self.resolution
        # from xy (cartesian) to uv coordinate
        if self._clipper_cfg.loc_origin == "lower":
            us = x // self.mpp_resolution #horizontal
            vs = H * np.ones_like(y) - y // self.mpp_resolution #vertical
        elif self._clipper_cfg.loc_origin == "upper":
            us = x // self.mpp_resolution
            vs = y // self.mpp_resolution
        elif self._clipper_cfg.loc_origin == "center":
            us = W //2 + x // self.mpp_resolution
            vs = H//2 + y // self.mpp_resolution
        for u, v in zip(us, vs):
            u = int(u)
            v = int(v)
            roll = self.slope_y[v, u]
            pitch = self.slope_x[v, u]
            yaw = 0.0
            q = quaternion.as_float_array(quaternion.from_euler_angles([roll, pitch, yaw]))
            quat.append(q)

        return np.stack(quat)

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