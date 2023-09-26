import numpy as np
from .Types import *
from .Samplers import *
from .Clippers import *
from .math_utils import rpy2quat

import copy

class BaseLayer:
    """
    Base class for all layers.
    A layer defines a geometric primitive to sample from.
    It takes a layer configuration and a sampler configuration as input."""

    def __init__(self, layer_cfg: Layer_T, sampler_cfg: Sampler_T, **kwarg) -> None:
        """
        Args:
            layer_cfg (Layer_T): The layer configuration.
            sampler_cfg (Sampler_T): The sampler configuration."""
        
        assert (self.__class__.__name__[:-5] == layer_cfg.__class__.__name__[:-2]), "Configuration mismatch. The type of the config must match the type of the layer."
        self._randomizer = None
        self._randomization_space = None

        self._sampler_cfg = copy.copy(sampler_cfg)
        self._layer_cfg = copy.copy(layer_cfg)

        if self._layer_cfg.output_space == self._sampler_cfg.randomization_space:
            self._skip_projection = True
        else:
            self._skip_projection = False

        if self._layer_cfg.transform is None:
            self._skip_transform = True
        else:
            self._skip_transform = False
            if isinstance(self._layer_cfg.transform, Transformation2D_T):
                assert self._layer_cfg.output_space == 2, "The output_shape must be equal to 2 to apply a 2D transform."
                self._T = self.buildTransform2D(self._layer_cfg.transform)
            else:
                assert self._layer_cfg.output_space == 3, "The output_shape must be equal to 3 to apply a 3D transform."
                self._T = self.buildTransform3D(self._layer_cfg.transform)
        
        self.getBounds()

    def initializeSampler(self) -> None:
        """
        Initializes the sampler."""

        self._sampler = Sampler_Factory.get(self._sampler_cfg)

    def getBounds(self) -> None:
        """
        Computes the bounds of the layer."""

        raise NotImplementedError()

    @staticmethod
    def buildRotMat3DfromQuat(xyzw: list) -> np.ndarray([3,3], dtype=float):
        """
        Builds a rotation matrix from a quaternion.

        Args:
            xyzw (list): The quaternion.

        Returns:
            np.ndarray([3,3], dtype=float): The rotation matrix."""

        q0 = xyzw[-1]
        q1 = xyzw[0]
        q2 = xyzw[1]
        q3 = xyzw[2]
        return 2*np.array([[q0*q0 + q1*q1, q1*q2 - q0*q3, q1*q3 + q0*q2],
                           [q1*q2 + q0*q3, q0*q0 + q2*q2, q2*q3 - q0*q1],
                           [q1*q3 - q0*q2, q2*q3 + q0*q1, q0*q0 + q3*q3]]) - np.eye(3)

    @staticmethod
    def buildRotMat2D(theta: float) -> np.ndarray([2,2], dtype=float):
        """
        Builds a rotation matrix from an angle.
        
        Args:
            theta (float): The angle.
            
        Returns:
            np.ndarray([2,2], dtype=float): The rotation matrix."""
        
        return np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])

    @staticmethod
    def buildRotMat3DfromEuler(xyz: list) -> np.ndarray([3,3], dtype=float):
        """
        Builds a rotation matrix from euler angles.
        
        Args:
            xyz (list): The euler angles. Angles are given in the following order: x, y, z.
            
        Returns:
            np.ndarray([3,3], dtype=float): The rotation matrix."""
        
        cx = np.cos(xyz[0])
        sx = np.cos(xyz[0])
        cy = np.cos(xyz[1])
        sy = np.cos(xyz[1])
        cz = np.cos(xyz[2])
        sz = np.cos(xyz[2])
        return np.array([[cy*cz, sx*sy*cz - cx*sz, cx*sy*cz + sx*sy],
                         [cy*sz, sx*sy*sz + cx*cz, cx*sy*sz - sx*cz],
                         [-sy, sx*cy, cx*cy]])

    def buildTransform3D(self, trans: Transformation3D_T) -> np.ndarray([4,4], dtype=float):
        """
        Builds a 3D transformation matrix from a Transformation3D_T object.
        
        Args:
            trans (Transformation3D_T): The transformation.
        
        Returns:
            np.ndarray([4,4], dtype=float): The transformation matrix."""
        
        t = trans.translation
        T = np.zeros([4,4])
        if isinstance(self._layer_cfg.transform.orientation, Quaternion_T):    
            q = trans.orientation
            T[:3,:3] = self.buildRotMat3DfromQuat([q.x, q.y, q.z, q.w])
        else:
            euler = trans.orientation
            T[:3,:3] = self.buildRotMat3DfromEuler([euler.x, euler.y, euler.z])

        T[3,3] = 1
        T[:3,3] = np.array([t.x, t.y, t.z]) 
        return T
    
    def buildTransform2D(self, trans: Transformation2D_T) -> np.ndarray([3,3], dtype=float):
        """
        Builds a 2D transformation matrix from a Transformation2D_T object.
        
        Args:
            trans (Transformation2D_T): The transformation.
        
        Returns:
            np.ndarray([3,3], dtype=float): The transformation matrix."""

        t = trans.translation
        T = np.zeros([3,3])
        theta = trans.orientation.theta
        T[:2,:2] = self.buildRotMat2D(theta)
        T[2,2] = 1
        T[:2,2] = np.array([t.x, t.y]) 
        return T
    
    def project(self, points: np.ndarray([])) -> np.ndarray([]):
        """
        Projects the points to the output space.
        
        Args:
            points (np.ndarray([])): The points to project.
            
        Returns:
            np.ndarray([]): The projected points."""
        
        if self._skip_projection:
            return points
        else:
            zeros = np.zeros([points.shape[0],self._layer_cfg.output_space - self._sampler_cfg.randomization_space])
            points = np.concatenate([points,zeros],axis=-1)
            return points

    def transform(self, points: np.ndarray([])) -> np.ndarray([]):
        """
        Transforms the points. (applies the transformation matrix)
        
        Args:
            points (np.ndarray([])): The points to transform.
            
        Returns:
            np.ndarray([]): The transformed points."""
        
        if self._skip_transform:
            return points
        else:
            ones = np.ones([points.shape[0],1])
            points = np.concatenate([points,ones],axis=-1)
            proj = np.matmul(self._T,points.T).T[:,:-1]
            return proj

    def sample(self, num: int = 1, **kwargs):
        """
        Samples points from the layer.
        
        Args:
            num (int, optional): The number of points to sample. Defaults to 1."""
        
        raise NotImplementedError()

    def applyProjection(self, points:np.ndarray([])) -> np.ndarray([]):
        """
        Applies the projection to the points.
        
        Args:
            points (np.ndarray([])): The points to project.
        
        Returns:
            np.ndarray([]): The projected points."""
        
        if self._skip_projection:
            return points
        else:
            return self.project(points)

    def applyTransform(self, points:np.ndarray([])) -> np.ndarray([]):
        """
        Applies the transformation to the points.
        
        Args:
            points (np.ndarray([])): The points to transform.
        
        Returns:
            np.ndarray([]): The transformed points."""
        
        if self._skip_transform:
            return points
        else:
            return self.transform(points)

    def __call__(self, num: int = 1, **kwargs) -> np.ndarray([]):
        """
        Samples points from the layer, projects them and transforms them.
        
        Args:
            num (int, optional): The number of points to sample. Defaults to 1."""

        points = self.sample(num = num, **kwargs)
        points = self.project(points)
        points = self.transform(points)
        return points

class Layer1D(BaseLayer):
    """
    Base layer for 1D primitives"""

    def __init__(self, layer_cfg: Layer_T, sampler_cfg: Sampler_T) -> None:
        super().__init__(layer_cfg, sampler_cfg)

class Layer2D(BaseLayer):
    """
    Base layer for 2D primitives"""

    def __init__(self, output_space: Layer_T, sampler_cfg: Sampler_T) -> None:
        super().__init__(output_space, sampler_cfg)

class Layer3D(BaseLayer):
    """
    Base layer for 3D primitives"""

    def __init__(self, output_space: Layer_T, sampler_cfg: Sampler_T) -> None:
        super().__init__(output_space, sampler_cfg)

class Layer4D(BaseLayer):
    """
    Base layer for 4D primitives"""

    def __init__(self, output_space: Layer_T, sampler_cfg: Sampler_T) -> None:
        super().__init__(output_space, sampler_cfg)

class ImageLayer(Layer2D):
    """
    A layer that creates something similar to a plane, but uses an image as 
    But, do not specify bounds, since the bounds are determined by the height and width of the image."""

    def __init__(self, layer_cfg: Image_T, sampler_cfg: Sampler_T) -> None:
        super().__init__(layer_cfg, sampler_cfg)
        self.initializeSampler()
        self._sampler.setMask(layer_cfg.data, layer_cfg.mpp_resolution)
        self._sampler._check_fn = self.checkBoundaries

    #def getBounds(self) -> None:
    #    """
    #    Computes the bounds of the layer."""

    #    self._bounds = None
    def getBounds(self) -> None:
        """
        Computes the bounds of the layer."""
        W,H = self._layer_cfg.data.shape
        self.xmin = 0
        self.ymin = 0
        self.xmax = W * self._layer_cfg.mpp_resolution
        self.ymax = H * self._layer_cfg.mpp_resolution
        self._bounds = np.array([[0, self.xmax],
                                [0, self.ymax]])
        

    def checkBoundaries(self, points: np.ndarray([])) -> np.ndarray([]):
        """
        Checks if the points are within the boundaries of the layer.
        
        Args:
            points (np.ndarray([])): The points to check.
        
        Returns:
            np.ndarray([]): A boolean array indicating if the points are within the boundaries of the layer."""
        
        b1 = points[:,0] >= self.xmin
        b2 = points[:,0] <= self.xmax
        b3 = points[:,1] >= self.ymin
        b4 = points[:,1] <= self.ymax
        return b1*b2*b3*b4

    def sample(self, num: int = 1, **kwargs) -> np.ndarray([]):
        """
        Samples points from the layer.
        
        Args:
            num (int, optional): The number of points to sample. Defaults to 1.
        
        Returns:
            np.ndarray([]): The sampled points."""

        return self._sampler.sample_image(num=num, bounds=self._bounds, **kwargs)

class NormalMapLayer(Layer4D):
    """
    Class which is similar to LineLayer
    But, do not specify bounds, since the bounds are determined by the height and width of the image."""

    def __init__(self, layer_cfg: Layer_T, sampler_cfg: Sampler_T) -> None:
        super().__init__(layer_cfg, sampler_cfg)
        self.initializeSampler()
    
    def getBounds(self) -> None:
        """
        Computes the bounds of the layer."""

        self._bounds = None

    def sample(self, query_point: np.ndarray, num: int = 1, **kwargs):
        """
        Samples points from the layer.
        
        Args:
            query_point (np.ndarray): The points at which the orientation should be taken.
            num (int, optional): The number of points to sample. Defaults to 1.
        
        Returns:
            np.ndarray([]): The sampled points."""    
        
        return self._sampler(num=num, bounds=self._bounds, query_point=query_point)
    
    def __call__(self, query_point: np.ndarray, num: int = 1, **kwargs) -> np.ndarray([]):
        """
        Samples points from the layer, projects them and transforms them."""

        points = self.sample(num = num, query_point=query_point, **kwargs)
        points = self.project(points)
        points = self.transform(points)
        return points

class LineLayer(Layer1D):
    """
    A layer that creates a Line primitives.
    When sampling on this layer, the points will be distributed on a line."""

    def __init__(self, layer_cfg: Line_T, sampler_cfg: Sampler_T) -> None:
        """
        Args:
            layer_cfg (Line_T): The layer configuration.
            sampler_cfg (Sampler_T): The sampler configuration."""
        
        super().__init__(layer_cfg, sampler_cfg)

        if isinstance(self._sampler_cfg, UniformSampler_T) or isinstance(self._sampler_cfg, LinearInterpolationSampler_T):
            self._sampler_cfg.randomization_space = 1
            self._sampler_cfg.min = [self._layer_cfg.xmin]
            self._sampler_cfg.max = [self._layer_cfg.xmax]

        self.initializeSampler()
        self._sampler._check_fn = self.checkBoundaries

    def checkBoundaries(self, points: np.ndarray([])) -> np.ndarray([]):
        """
        Checks if the points are within the boundaries of the layer.
        
        Args:
            points (np.ndarray([])): The points to check.
            
        Returns:
            np.ndarray([]): A boolean array indicating if the points are within the boundaries of the layer."""
        
        b1 = points[:,0] >= self._layer_cfg.xmin
        b2 = points[:,0] <= self._layer_cfg.xmax
        return b1*b2

    def getBounds(self) -> None:
        """
        Computes the bounds of the layer."""

        self._bounds = np.array([[self._layer_cfg.xmin, self._layer_cfg.xmax]])

    def createMask(self) -> None:
        pass

    def sample(self, num: int = 1, **kwargs) -> np.ndarray([]):
        """
        Samples points from the layer.
        
        Args:
            num (int, optional): The number of points to sample. Defaults to 1.
            
        Returns:
            np.ndarray([]): The sampled points."""
        
        return self._sampler(num=num, bounds=self._bounds, **kwargs)

class CircleLayer(Layer1D):
    """
    A layer that creates a Circle primitives.
    When sampling on this layer, the points will be distributed on a circle."""

    def __init__(self, layer_cfg: Circle_T, sampler_cfg: Sampler_T) -> None:
        """
        Args:
            layer_cfg (Circle_T): The layer configuration.
            sampler_cfg (Sampler_T): The sampler configuration."""
        
        super().__init__(layer_cfg, sampler_cfg)

        if isinstance(self._sampler_cfg, UniformSampler_T) or isinstance(self._sampler_cfg, LinearInterpolationSampler_T):
            self._sampler_cfg.randomization_space = 1
            self._sampler_cfg.min = [self._layer_cfg.theta_min]
            self._sampler_cfg.max = [self._layer_cfg.theta_max]

        self.initializeSampler()
        self._sampler._check_fn = self.checkBoundaries

        if self._layer_cfg.output_space == (self._sampler_cfg.randomization_space + 1):
            self._skip_projection = True
        else:
            self._skip_projection = False

    def checkBoundaries(self, points: np.ndarray([])) -> np.ndarray([]):
        """
        Checks if the points are within the boundaries of the layer.
        
        Args:
            points (np.ndarray([])): The points to check.
            
        Returns:
            np.ndarray([]): A boolean array indicating if the points are within the boundaries of the layer."""

        b1 = points[:,0] >= self._layer_cfg.theta_min
        b2 = points[:,0] <= self._layer_cfg.theta_max
        return b1*b2

    def getBounds(self) -> None:
        """
        Computes the bounds of the layer."""

        self._bounds = np.array([[self._layer_cfg.theta_min, self._layer_cfg.theta_max]])

    def createMask(self) -> None:
        pass

    def sample(self, num: int = 1, **kwargs) -> np.ndarray([]):
        """
        Samples points from the layer.
        Projects from polar coordinates to x,y coordinates.
        
        Args:
            num (int, optional): The number of points to sample. Defaults to 1.
            
        Returns:
            np.ndarray([]): The sampled points."""

        theta = self._sampler(num=num, bounds=self._bounds, **kwargs)
        x = self._layer_cfg.center[0] + np.cos(theta)*self._layer_cfg.radius*self._layer_cfg.alpha
        y = self._layer_cfg.center[1] + np.sin(theta)*self._layer_cfg.radius*self._layer_cfg.beta
        return np.stack([x,y]).T[0]

    def project(self, points: np.ndarray([])) -> np.ndarray([]):
        """
        Projects the points to the output space.
        
        Args:
            points (np.ndarray([])): The points to project.
        
        Returns:
            np.ndarray([]): The projected points."""

        if self._skip_projection:
            return points
        else:
            zeros = np.zeros([points.shape[0],self._layer_cfg.output_space - self._sampler_cfg.randomization_space -1])
            points = np.concatenate([points,zeros],axis=-1)
            return points

class PlaneLayer(Layer2D):
    """
    A layer that creates a Plane primitives.
    When sampling on this layer, the points will be distributed on a plane."""

    def __init__(self, layer_cfg: Plane_T, sampler_cfg: Sampler_T) -> None:
        """
        Args:
            layer_cfg (Plane_T): The layer configuration.
            sampler_cfg (Sampler_T): The sampler configuration."""
        
        super().__init__(layer_cfg, sampler_cfg)

        if isinstance(self._sampler_cfg, UniformSampler_T) or isinstance(self._sampler_cfg, LinearInterpolationSampler_T):
            self._sampler_cfg.randomization_space = 2
            self._sampler_cfg.min = [self._layer_cfg.xmin, self._layer_cfg.ymin]
            self._sampler_cfg.max = [self._layer_cfg.xmax, self._layer_cfg.ymax]

        self.initializeSampler()
        self._sampler._check_fn = self.checkBoundaries


    def checkBoundaries(self, points: np.ndarray([])) -> np.ndarray([]):
        """
        Checks if the points are within the boundaries of the layer.
        
        Args:
            points (np.ndarray([])): The points to check.
        
        Returns:
            np.ndarray([]): A boolean array indicating if the points are within the boundaries of the layer."""
        
        b1 = points[:,0] >= self._layer_cfg.xmin
        b2 = points[:,0] <= self._layer_cfg.xmax
        b3 = points[:,1] >= self._layer_cfg.ymin
        b4 = points[:,1] <= self._layer_cfg.ymax
        return b1*b2*b3*b4

    def getBounds(self) -> None:
        """
        Computes the bounds of the layer."""

        self._bounds = np.array([[self._layer_cfg.xmin, self._layer_cfg.xmax],
                                [self._layer_cfg.ymin, self._layer_cfg.ymax]])

    def createMask(self) -> None:
        pass

    def sample(self, num: int = 1, **kwargs) -> np.ndarray([]):
        """
        Samples points from the layer.
        
        Args:
            num (int, optional): The number of points to sample. Defaults to 1.
            
        Returns:
            np.ndarray([]): The sampled points."""
        
        return self._sampler(num=num, bounds=self._bounds, **kwargs)

class DiskLayer(Layer2D):
    """
    A layer that creates a Disk primitives.
    When sampling on this layer, the points will be distributed on a disk."""

    def __init__(self, layer_cfg: Disk_T, sampler_cfg: Sampler_T) -> None:
        """
        Args:
            layer_cfg (Disk_T): The layer configuration.
            sampler_cfg (Sampler_T): The sampler configuration."""
        
        super().__init__(layer_cfg, sampler_cfg)

        if isinstance(self._sampler_cfg, UniformSampler_T) or isinstance(self._sampler_cfg, LinearInterpolationSampler_T):
            self._sampler_cfg.randomization_space = 2
            self._sampler_cfg.min = [0.0, 0.0]
            self._sampler_cfg.max = [1.0, 1.0]
        elif isinstance(self._sampler_cfg, NormalSampler_T):
            tmp1 = (self._sampler_cfg.mean[0] - self._layer_cfg.radius_min) / (self._layer_cfg.radius_max - self._layer_cfg.radius_min)
            tmp2 = (self._sampler_cfg.mean[1] - self._layer_cfg.theta_min) / (self._layer_cfg.theta_max - self._layer_cfg.theta_min)
            self._sampler_cfg.mean = np.array([tmp1, tmp2])
            tmp1 = self._sampler_cfg.std[0,0] / (self._layer_cfg.radius_max - self._layer_cfg.radius_min)
            tmp2 = self._sampler_cfg.std[1,1] / (self._layer_cfg.theta_max - self._layer_cfg.theta_min)
            self._sampler_cfg.std = np.array([[tmp1, 0],[0, tmp2]])

        self.initializeSampler()
        self._sampler._check_fn = self.checkBoundaries

    def checkBoundaries(self, points: np.ndarray([])) -> np.ndarray([]):
        """
        Checks if the points are within the boundaries of the layer.
        
        Args:
            points (np.ndarray([])): The points to check.
        
        Returns:
            np.ndarray([]): A boolean array indicating if the points are within the boundaries of the layer."""
        
        b1 = points[:,0] >= 0.0
        b2 = points[:,0] <= 1.0
        b3 = points[:,1] >= 0.0
        b4 = points[:,1] <= 1.0
        return b1*b2*b3*b4

    def getBounds(self) -> None:
        """
        Computes the bounds of the layer."""

        self._bounds = np.array([[0.0, 1.0],
                                [0.0, 1.0]])
        self._area = (self._layer_cfg.theta_max - self._layer_cfg.theta_min) * (self._layer_cfg.radius_max - self._layer_cfg.radius_min)**2

    def createMask(self) -> None:
        pass

    def sample(self, num: int = 1, **kwargs) -> np.ndarray([]):
        """
        Samples points from the layer.
        
        Args:
            num (int, optional): The number of points to sample. Defaults to 1.
            
        Returns:
            np.ndarray([]): The sampled points."""
        
        rand = self._sampler(num=num, bounds=self._bounds, area=self._area, **kwargs)

        r_rescaled = (self._layer_cfg.radius_min + rand[:,0] * (self._layer_cfg.radius_max - self._layer_cfg.radius_min)) / self._layer_cfg.radius_max
        r = np.sqrt(r_rescaled) * self._layer_cfg.radius_max
        t = self._layer_cfg.theta_min + rand[:,1] * (self._layer_cfg.theta_max - self._layer_cfg.theta_min)

        x = self._layer_cfg.center[0] + np.cos(t)*r*self._layer_cfg.alpha
        y = self._layer_cfg.center[1] + np.sin(t)*r*self._layer_cfg.beta
        return np.stack([x,y]).T

class CubeLayer(Layer3D):
    """
    A layer that creates a Cube primitives.
    When sampling on this layer, the points will be distributed on a cube."""

    def __init__(self, layer_cfg: Cube_T, sampler_cfg: Sampler_T) -> None:
        """
        Args:
            layer_cfg (Cube_T): The layer configuration.
            sampler_cfg (Sampler_T): The sampler configuration.""" 
        
        super().__init__(layer_cfg, sampler_cfg)

        if isinstance(self._sampler_cfg, UniformSampler_T) or isinstance(self._sampler_cfg, LinearInterpolationSampler_T):
            self._sampler_cfg.randomization_space = 3
            self._sampler_cfg.min = [self._layer_cfg.xmin, self._layer_cfg.ymin, self._layer_cfg.zmin]
            self._sampler_cfg.max = [self._layer_cfg.xmax, self._layer_cfg.ymax, self._layer_cfg.zmax]

        self.initializeSampler()
        self._sampler._check_fn = self.checkBoundaries

    def checkBoundaries(self, points: np.ndarray([])) -> np.ndarray([]):
        """
        Checks if the points are within the boundaries of the layer.
        
        Args:
            points (np.ndarray([])): The points to check.
        
        Returns:
            np.ndarray([]): A boolean array indicating if the points are within the boundaries of the layer."""
        
        b1 = points[:,0] >= self._layer_cfg.xmin
        b2 = points[:,0] <= self._layer_cfg.xmax
        b3 = points[:,1] >= self._layer_cfg.ymin
        b4 = points[:,1] <= self._layer_cfg.ymax
        b5 = points[:,2] >= self._layer_cfg.zmin
        b6 = points[:,2] <= self._layer_cfg.zmax
        return b1*b2*b3*b4*b5*b6

    def getBounds(self) -> None:
        """
        Computes the bounds of the layer."""

        self._bounds = np.array([[self._layer_cfg.xmin, self._layer_cfg.xmax],
                                [self._layer_cfg.ymin, self._layer_cfg.ymax],
                                [self._layer_cfg.zmin, self._layer_cfg.zmax]])

    def createMask(self) -> None:
        pass

    def sample(self, num:int =1, **kwargs) -> np.ndarray([]):
        """
        Samples points from the layer.
        
        Args:
            num (int, optional): The number of points to sample. Defaults to 1.
        
        Returns:
            np.ndarray([]): The sampled points."""
        
        return self._sampler(num=num, bounds=self._bounds, **kwargs)

class RollPitchYawLayer(Layer4D):
    """
    A layer that creates a RollPitchYaw primitives.
    When sampling on this layer, the points will be distributed on a RollPitchYaw.
    The quaternion should not be sampled the way it is sampled here. It's just happens to be the easiest method I can think of."""

    def __init__(self, layer_cfg: RollPitchYaw_T, sampler_cfg: Sampler_T) -> None:
        """
        Args:
            layer_cfg (RollPitchYaw_T): The layer configuration.
            sampler_cfg (Sampler_T): The sampler configuration."""
        
        super().__init__(layer_cfg, sampler_cfg)

        if isinstance(self._sampler_cfg, UniformSampler_T) or isinstance(self._sampler_cfg, LinearInterpolationSampler_T):
            self._sampler_cfg.randomization_space = 3
            self._sampler_cfg.min = [self._layer_cfg.rmin, self._layer_cfg.pmin, self._layer_cfg.ymin]
            self._sampler_cfg.max = [self._layer_cfg.rmax, self._layer_cfg.pmax, self._layer_cfg.ymax]

        self.initializeSampler()
        self._sampler._check_fn = self.checkBoundaries

    def checkBoundaries(self, points: np.ndarray([])) -> np.ndarray([]):
        """
        Checks if the points are within the boundaries of the layer.
        
        Args:
            points (np.ndarray([])): The points to check.
        
        Returns:
            np.ndarray([]): A boolean array indicating if the points are within the boundaries of the layer."""
        
        b1 = points[:,0] >= self._layer_cfg.rmin
        b2 = points[:,0] <= self._layer_cfg.rmax
        b3 = points[:,1] >= self._layer_cfg.pmin
        b4 = points[:,1] <= self._layer_cfg.pmax
        b5 = points[:,2] >= self._layer_cfg.ymin
        b6 = points[:,2] <= self._layer_cfg.ymax
        return b1*b2*b3*b4*b5*b6

    def getBounds(self) -> None:
        """
        Computes the bounds of the layer."""

        self._bounds = np.array([[self._layer_cfg.rmin, self._layer_cfg.rmax],
                                [self._layer_cfg.pmin, self._layer_cfg.pmax],
                                [self._layer_cfg.ymin, self._layer_cfg.ymax]])

    def createMask(self) -> None:
        pass

    def sample(self, num:int =1, **kwargs) -> np.ndarray([]):
        """
        Samples points from the layer.
        
        Args:
            num (int, optional): The number of points to sample. Defaults to 1.
            
        Returns:
            np.ndarray([]): The sampled points."""

        rand = self._sampler(num=num, bounds=self._bounds, **kwargs)
        # Batch RPY 2 Quat
        quat = rpy2quat(rand)
        return quat


class SphereLayer(Layer3D):
    """
    A layer that creates a Sphere primitives.
    When sampling on this layer, the points will be distributed on a sphere."""

    def __init__(self, layer_cfg: Sphere_T, sampler_cfg: Sampler_T) -> None:
        """
        Args:
            layer_cfg (Sphere_T): The layer configuration.
            sampler_cfg (Sampler_T): The sampler configuration."""
        
        super().__init__(layer_cfg, sampler_cfg)

        if isinstance(self._sampler_cfg, UniformSampler_T) or isinstance(self._sampler_cfg, LinearInterpolationSampler_T):
            self._sampler_cfg.randomization_space = 3
            self._sampler_cfg.min = [0,0,0]
            self._sampler_cfg.max = [1,1,1]
        elif isinstance(self._sampler_cfg, NormalSampler_T):
            tmp1 = (self._sampler_cfg.mean[0] - self._layer_cfg.radius_min) / (self._layer_cfg.radius_max - self._layer_cfg.radius_min)
            tmp2 = (self._sampler_cfg.mean[1] - self._layer_cfg.phi_min) / (self._layer_cfg.phi_max - self._layer_cfg.phi_min)
            tmp3 = (self._sampler_cfg.mean[2] - self._layer_cfg.theta_min) / (self._layer_cfg.theta_max - self._layer_cfg.theta_min)
            self._sampler_cfg.mean = np.array([tmp1, tmp2, tmp3])
            tmp1 = self._sampler_cfg.std[0,0] / (self._layer_cfg.radius_max - self._layer_cfg.radius_min)
            tmp2 = self._sampler_cfg.std[1,1] / (self._layer_cfg.phi_max - self._layer_cfg.phi_min)
            tmp3 = self._sampler_cfg.std[2,2] / (self._layer_cfg.theta_max - self._layer_cfg.theta_min)
            self._sampler_cfg.std = np.array([[tmp1, 0, 0],[0, tmp2, 0],[0, 0, tmp3]])

        self.initializeSampler()
        self._sampler._check_fn = self.checkBoundaries

    def checkBoundaries(self, points: np.ndarray([])) -> np.ndarray([]):
        """
        Checks if the points are within the boundaries of the layer.
        
        Args:
            points (np.ndarray([])): The points to check.
            
        Returns:
            np.ndarray([]): A boolean array indicating if the points are within the boundaries of the layer."""
        
        b1 = points[:,0] >= 0.0
        b2 = points[:,0] <= 1.0
        b3 = points[:,1] >= 0.0
        b4 = points[:,1] <= 1.0
        b5 = points[:,2] >= 0.0
        b6 = points[:,2] <= 1.0
        return b1*b2*b3*b4*b5*b6

    def getBounds(self) -> None:
        """
        Computes the bounds of the layer."""

        self._bounds = np.array([[0.0, 1.0],
                                [0.0, 1.0],
                                [0.0, 1.0]])
        self._area = (self._layer_cfg.theta_max - self._layer_cfg.theta_min) * (self._layer_cfg.phi_max - self._layer_cfg.phi_min) * (self._layer_cfg.radius_max - self._layer_cfg.radius_min)**2

    def createMask(self) -> None:
        pass

    def sample(self, num:int = 1, **kwargs) -> np.ndarray([]):
        """
        Samples points from the layer.
        
        Args:
            num (int, optional): The number of points to sample. Defaults to 1.
        
        Returns:
            np.ndarray([]): The sampled points."""
        
        rand = self._sampler(num=num, bounds=self._bounds, area=self._area, **kwargs)

        r_rescaled = (self._layer_cfg.radius_min + rand[:,0] * (self._layer_cfg.radius_max - self._layer_cfg.radius_min)) / self._layer_cfg.radius_max
        r = np.sqrt(r_rescaled) * self._layer_cfg.radius_max
        t = self._layer_cfg.theta_min + rand[:,1] * (self._layer_cfg.theta_max - self._layer_cfg.theta_min)
        p = self._layer_cfg.phi_min + rand[:,2] * (self._layer_cfg.phi_max - self._layer_cfg.phi_min)

        x = self._layer_cfg.center[0] + np.sin(p)*np.cos(t)*r*self._layer_cfg.alpha
        y = self._layer_cfg.center[1] + np.sin(p)*np.sin(t)*r*self._layer_cfg.beta
        z = self._layer_cfg.center[2] + np.cos(p)*r*self._layer_cfg.ceta
        return np.stack([x,y,z]).T

class CylinderLayer(Layer3D):
    """
    A layer that creates a Cylinder primitives.
    When sampling on this layer, the points will be distributed on a cylinder."""

    def __init__(self, layer_cfg: Cylinder_T, sampler_cfg: Sampler_T) -> None:
        """
        Args:
            layer_cfg (Cylinder_T): The layer configuration.
            sampler_cfg (Sampler_T): The sampler configuration."""
        
        super().__init__(layer_cfg, sampler_cfg)

        if isinstance(self._sampler_cfg, UniformSampler_T) or isinstance(self._sampler_cfg, LinearInterpolationSampler_T):
            self._sampler_cfg.randomization_space = 3
            self._sampler_cfg.min = [0.0, 0.0, 0.0]
            self._sampler_cfg.max = [1.0, 1.0, 1.0]
        elif isinstance(self._sampler_cfg, NormalSampler_T):
            tmp1 = (self._sampler_cfg.mean[0] - self._layer_cfg.radius_min) / (self._layer_cfg.radius_max - self._layer_cfg.radius_min)
            tmp2 = (self._sampler_cfg.mean[1] - self._layer_cfg.height_min) / (self._layer_cfg.height_max - self._layer_cfg.height_min)
            tmp3 = (self._sampler_cfg.mean[2] - self._layer_cfg.theta_min) / (self._layer_cfg.theta_max - self._layer_cfg.theta_min)
            self._sampler_cfg.mean = np.array([tmp1, tmp2, tmp3])
            tmp1 = self._sampler_cfg.std[0,0] / (self._layer_cfg.radius_max - self._layer_cfg.radius_min)
            tmp2 = self._sampler_cfg.std[1,1] / (self._layer_cfg.height_max - self._layer_cfg.height_min)
            tmp3 = self._sampler_cfg.std[2,2] / (self._layer_cfg.theta_max - self._layer_cfg.theta_min)
            self._sampler_cfg.std = np.array([[tmp1, 0, 0],[0, tmp2, 0],[0, 0, tmp3]])

        self.initializeSampler()
        self._sampler._check_fn = self.checkBoundaries

    def checkBoundaries(self, points: np.ndarray([])) -> np.ndarray([]):
        """
        Checks if the points are within the boundaries of the layer.
        
        Args:
            points (np.ndarray([])): The points to check.
            
        Returns:
            np.ndarray([]): A boolean array indicating if the points are within the boundaries of the layer."""
        
        b1 = points[:,0] >= 0.0
        b2 = points[:,0] <= 1.0
        b3 = points[:,1] >= 0.0
        b4 = points[:,1] <= 1.0
        b5 = points[:,2] >= 0.0
        b6 = points[:,2] <= 1.0
        return b1*b2*b3*b4*b5*b6

    def getBounds(self) -> None:
        """
        Computes the bounds of the layer."""

        self._bounds = np.array([[0.0, 1.0],
                                [0.0, 1.0],
                                [0.0, 1.0]])
        self._area = (self._layer_cfg.theta_max - self._layer_cfg.theta_min) * (self._layer_cfg.height_max - self._layer_cfg.height_min) * (self._layer_cfg.radius_max - self._layer_cfg.radius_min)**2

    def createMask(self) -> None:
        pass

    def sample(self, num: int = 1, **kwargs) -> np.ndarray([]):
        """
        Samples points from the layer.
        
        Args:
            num (int, optional): The number of points to sample. Defaults to 1.
            
        Returns:
            np.ndarray([]): The sampled points."""

        rand = self._sampler(num=num, bounds=self._bounds, area=self._area, **kwargs)

        r_rescaled = (self._layer_cfg.radius_min + rand[:,0] * (self._layer_cfg.radius_max - self._layer_cfg.radius_min)) / self._layer_cfg.radius_max
        r = np.sqrt(r_rescaled) * self._layer_cfg.radius_max
        h = self._layer_cfg.height_min + (self._layer_cfg.height_max - self._layer_cfg.height_min)*rand[:,1]
        theta = np.pi*2*rand[:,2]

        x = self._layer_cfg.center[0] + np.cos(theta)*r*self._layer_cfg.alpha
        y = self._layer_cfg.center[1] + np.sin(theta)*r*self._layer_cfg.beta
        z = h
        return np.stack([x,y,z]).T

class ConeLayer(Layer3D):
    """
    A layer that creates a Cone primitives.
    When sampling on this layer, the points will be distributed on a cone."""

    def __init__(self, layer_cfg: Cone_T, sampler_cfg: Sampler_T) -> None:
        """
        Args:
            layer_cfg (Cone_T): The layer configuration.
            sampler_cfg (Sampler_T): The sampler configuration."""
        
        super().__init__(layer_cfg, sampler_cfg)

        if isinstance(self._sampler_cfg, UniformSampler_T) or isinstance(self._sampler_cfg, LinearInterpolationSampler_T):
            self._sampler_cfg.randomization_space = 3
            self._sampler_cfg.min = [0.0, 0.0, 0.0]
            self._sampler_cfg.max = [1.0, 1.0, 1.0]
        elif isinstance(self._sampler_cfg, NormalSampler_T):
            tmp1 = (self._sampler_cfg.mean[0] - self._layer_cfg.radius_min) / (self._layer_cfg.radius_max - self._layer_cfg.radius_min)
            tmp2 = (self._sampler_cfg.mean[1] - self._layer_cfg.height_min) / (self._layer_cfg.height_max - self._layer_cfg.height_min)
            tmp3 = (self._sampler_cfg.mean[2] - self._layer_cfg.theta_min) / (self._layer_cfg.theta_max - self._layer_cfg.theta_min)
            self._sampler_cfg.mean = np.array([tmp1, tmp2, tmp3])
            tmp1 = self._sampler_cfg.std[0,0] / (self._layer_cfg.radius_max - self._layer_cfg.radius_min)
            tmp2 = self._sampler_cfg.std[1,1] / (self._layer_cfg.height_max - self._layer_cfg.height_min)
            tmp3 = self._sampler_cfg.std[2,2] / (self._layer_cfg.theta_max - self._layer_cfg.theta_min)
            self._sampler_cfg.std = np.array([[tmp1, 0, 0],[0, tmp2, 0],[0, 0, tmp3]])

        self.initializeSampler()
        self._sampler._check_fn = self.checkBoundaries

    def checkBoundaries(self, points: np.ndarray([])) -> np.ndarray([]):
        """
        Checks if the points are within the boundaries of the layer.
        
        Args:
            points (np.ndarray([])): The points to check.
        
        Returns:
            np.ndarray([]): A boolean array indicating if the points are within the boundaries of the layer."""
        
        b1 = points[:,0] >= 0.0
        b2 = points[:,0] <= 1.0
        b3 = points[:,1] >= 0.0
        b4 = points[:,1] <= 1.0
        b5 = points[:,2] >= 0.0
        b6 = points[:,2] <= 1.0
        return b1*b2*b3*b4*b5*b6

    def getBounds(self) -> None:
        """
        Computes the bounds of the layer."""

        self._bounds = np.array([[0.0, 1.0],
                                [0.0, 1.0],
                                [0.0, 1.0]])
        self._area = (1/6)*(self._layer_cfg.theta_max - self._layer_cfg.theta_min) * (self._layer_cfg.height_max - self._layer_cfg.height_min) * (self._layer_cfg.radius_max - self._layer_cfg.radius_min)**2

    def createMask(self) -> None:
        pass

    def sample(self, num: int = 1, **kwargs) -> np.ndarray([]):
        """
        Samples points from the layer.
        
        Args:
            num (int, optional): The number of points to sample. Defaults to 1.
        
        Returns:
            np.ndarray([]): The sampled points."""

        rand = self._sampler(num=num, bounds=self._bounds, area=self._area, **kwargs)

        r_rescaled = (self._layer_cfg.radius_min + rand[:,0] * (self._layer_cfg.radius_max - self._layer_cfg.radius_min)) / self._layer_cfg.radius_max
        r = np.sqrt(r_rescaled) * self._layer_cfg.radius_max
        h = np.power(rand[:,1],1/3)
        theta = np.pi*2*rand[:,2]

        x = self._layer_cfg.center[0] + np.cos(theta)*r*h*self._layer_cfg.alpha
        y = self._layer_cfg.center[1] + np.sin(theta)*r*h*self._layer_cfg.beta
        z = self._layer_cfg.center[2] + self._layer_cfg.height_min +  h*(self._layer_cfg.height_max - self._layer_cfg.height_min)
        return np.stack([x,y,z]).T

class TorusLayer(Layer3D):
    """
    A layer that creates a Torus primitives.
    When sampling on this layer, the points will be distributed on a torus."""

    def __init__(self, layer_cfg: Torus_T, sampler_cfg: Sampler_T) -> None:
        """
        Args:
            layer_cfg (Torus_T): The layer configuration.
            sampler_cfg (Sampler_T): The sampler configuration.""" 
        
        super().__init__(layer_cfg, sampler_cfg)

        if isinstance(self._sampler_cfg, UniformSampler_T) or isinstance(self._sampler_cfg, LinearInterpolationSampler_T):
            self._sampler_cfg.randomization_space = 3
            self._sampler_cfg.min = [0,0,0]
            self._sampler_cfg.max = [1,1,1]
        elif isinstance(self._sampler_cfg, NormalSampler_T):
            tmp1 = (self._sampler_cfg.mean[0] - self._layer_cfg.radius2_min) / (self._layer_cfg.radius2_max - self._layer_cfg.radius2_min)
            tmp2 = (self._sampler_cfg.mean[1] - self._layer_cfg.theta1_min) / (self._layer_cfg.theta1_max - self._layer_cfg.theta1_min)
            tmp3 = (self._sampler_cfg.mean[2] - self._layer_cfg.theta2_min) / (self._layer_cfg.theta2_max - self._layer_cfg.theta2_min)
            self._sampler_cfg.mean = np.array([tmp1, tmp2, tmp3])
            tmp1 = self._sampler_cfg.std[0,0] / (self._layer_cfg.radius2_max - self._layer_cfg.radius2_min)
            tmp2 = self._sampler_cfg.std[1,1] / (self._layer_cfg.theta1_max - self._layer_cfg.theta1_min)
            tmp3 = self._sampler_cfg.std[2,2] / (self._layer_cfg.theta2_max - self._layer_cfg.theta2_min)
            self._sampler_cfg.std = np.array([[tmp1, 0, 0],[0, tmp2, 0],[0, 0, tmp3]])

        self.initializeSampler()
        self._sampler._check_fn = self.checkBoundaries

    def checkBoundaries(self, points: np.ndarray([])) -> np.ndarray([]):
        """
        Checks if the points are within the boundaries of the layer.
        
        Args:
            points (np.ndarray([])): The points to check.
            
        Returns:
            np.ndarray([]): A boolean array indicating if the points are within the boundaries of the layer."""
        
        b1 = points[:,0] >= 0.0
        b2 = points[:,0] <= 1.0
        b3 = points[:,1] >= 0.0
        b4 = points[:,1] <= 1.0
        b5 = points[:,2] >= 0.0
        b6 = points[:,2] <= 1.0
        return b1*b2*b3*b4*b5*b6

    def getBounds(self) -> None:
        """
        Computes the bounds of the layer."""

        self._bounds = np.array([[0.0, 1.0],
                                [0.0, 1.0],
                                [0.0, 1.0]])

        self._area = (self._layer_cfg.theta1_max - self._layer_cfg.theta1_min) * (self._layer_cfg.theta2_max - self._layer_cfg.theta2_min) * (self._layer_cfg.radius1) * (self._layer_cfg.radius2_max - self._layer_cfg.radius2_min)**2


    def createMask(self) -> None:
        pass

    def sample(self, num: int = 1, **kwargs) -> np.ndarray([]):
        """
        Samples points from the layer.
        
        Args:
            num (int, optional): The number of points to sample. Defaults to 1.
        
        Returns:
            np.ndarray([]): The sampled points."""

        rand = self._sampler(num=num, bounds=self._bounds, area=self._area, **kwargs)

        r2_rescaled = (self._layer_cfg.radius2_min + rand[:,0] * (self._layer_cfg.radius2_max - self._layer_cfg.radius2_min)) / self._layer_cfg.radius2_max
        r2 = np.sqrt(r2_rescaled) * self._layer_cfg.radius2_max
        t = self._layer_cfg.theta1_min + rand[:,1] * (self._layer_cfg.theta1_max - self._layer_cfg.theta1_min)
        p = self._layer_cfg.theta2_min + rand[:,2] * (self._layer_cfg.theta2_max - self._layer_cfg.theta2_min)

        x = self._layer_cfg.center[0] + self._layer_cfg.radius1*np.cos(t) + r2*np.cos(t)*np.sin(p)*self._layer_cfg.alpha
        y = self._layer_cfg.center[1] + self._layer_cfg.radius1*np.sin(t) + r2*np.sin(t)*np.sin(p)*self._layer_cfg.beta
        z = self._layer_cfg.center[2] + np.cos(p)*r2*self._layer_cfg.ceta
        return np.stack([x,y,z]).T

class LayerFactory:
    """
    A factory class for creating layers."""

    def __init__(self):
        self.creators = {}
    
    def register(self, name: str, class_: BaseLayer) -> None:
        """
        Registers a layer.
        
        Args:
            name (str): The name of the layer.
            class_ (BaseLayer): The layer class."""
        self.creators[name] = class_
        
    def get(self, cfg: Layer_T, sampler_cfg: Sampler_T) -> BaseLayer:
        """
        Gets a layer.
        
        Args:
            cfg (Layer_T): The layer configuration.
            sampler_cfg (Sampler_T): The sampler configuration.
        
        Raises:
            ValueError: If the layer is not registered.
            
        Returns:
            BaseLayer: The layer."""

        if cfg.__class__.__name__ not in self.creators.keys():
            raise ValueError("Unknown layer requested.")
        return self.creators[cfg.__class__.__name__](cfg, sampler_cfg)

Layer_Factory = LayerFactory()
Layer_Factory.register("Line_T", LineLayer)
Layer_Factory.register("Circle_T", CircleLayer)
Layer_Factory.register("Plane_T", PlaneLayer)
Layer_Factory.register("Disk_T", DiskLayer)
Layer_Factory.register("Cube_T", CubeLayer)
Layer_Factory.register("Sphere_T", SphereLayer)
Layer_Factory.register("Cylinder_T", CylinderLayer)
Layer_Factory.register("Cone_T", ConeLayer)
Layer_Factory.register("Torus_T", TorusLayer)
Layer_Factory.register("Image_T", ImageLayer)
Layer_Factory.register("NormalMap_T", NormalMapLayer)
Layer_Factory.register("RollPitchYaw_T", RollPitchYawLayer)

#TODO: Implement these layers
#class Spline(Layer1D):
#    def __init__(self) -> None:
#        super().__init__()
#        # [[start, end]]
#        # Rotation matrix

#class SurfacePolygon(Layer2D):
#    def __init__(self) -> None:
#        super().__init__()


