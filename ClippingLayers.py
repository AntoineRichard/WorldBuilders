import numpy as np
from .Types import *
from .Samplers import *
from .Clippers import *
import copy

class BaseClippingLayer:
    def __init__(self, layer_cfg: Layer_T, sampler_cfg: Sampler_T, **kwarg) -> None:
        #assert (self.__class__.__name__[:-5] == layer_cfg.__class__.__name__[:-2]), "Configuration mismatch. The type of the config must match the type of the layer."
        self._randomizer = None
        self._randomization_space = None

        self._sampler_cfg = copy.copy(sampler_cfg)
        self._layer_cfg = copy.copy(layer_cfg)

    def initializeClipper(self) -> None:
        self._sampler = Clipper_Factory.get(self._sampler_cfg)

    def sample(self, num: int = 1):
        raise NotImplementedError()

    def __call__(self, num: int = 1) -> np.ndarray([]):
        points = self.sample(num = num)
        return points

class ClippingLayer1D(BaseClippingLayer):
    # Defines a 1D randomization space.
    def __init__(self, layer_cfg: Layer_T, sampler_cfg: Sampler_T) -> None:
        super().__init__(layer_cfg, sampler_cfg)

class ClippingLayer4D(BaseClippingLayer):
    def __init__(self, output_space: Layer_T, sampler_cfg: Sampler_T) -> None:
        super().__init__(output_space, sampler_cfg)

class ClippingImageLayer(ClippingLayer1D):
    """
    Class which is similar with LineLayer
    But, do not specify bound, since the bound is determined by height image.
    """
    def __init__(self, layer_cfg: Layer_T, sampler_cfg: Sampler_T) -> None:
        super().__init__(layer_cfg, sampler_cfg)
        self.initializeClipper()

    def sample(self, query_point: np.ndarray, num: int = 1):
        return self._sampler(num=num, query_point=query_point)
    
    def __call__(self, query_point: np.ndarray, num: int = 1) -> np.ndarray([]):
        points = self.sample(num = num, query_point=query_point)
        return points

class ClippingNormalMapLayer(ClippingLayer4D):
    def __init__(self, layer_cfg: Layer_T, sampler_cfg: Sampler_T) -> None:
        super().__init__(layer_cfg, sampler_cfg)
        self.initializeClipper()

    def sample(self, query_point: np.ndarray, num: int = 1):
        return self._sampler(num=num, query_point=query_point)
    
    def __call__(self, query_point: np.ndarray, num: int = 1) -> np.ndarray([]):
        points = self.sample(num = num, query_point=query_point)
        return points

class ClippingLayerFactory:
    def __init__(self):
        self.creators = {}
    
    def register(self, name: str, class_: BaseClippingLayer) -> None:
        self.creators[name] = class_
        
    def get(self, cfg: Layer_T, sampler_cfg: Sampler_T) -> BaseClippingLayer:
        if cfg.__class__.__name__ not in self.creators.keys():
            raise ValueError("Unknown layer requested.")
        return self.creators[cfg.__class__.__name__](cfg, sampler_cfg)

Clipping_Layer_Factory = ClippingLayerFactory()
Clipping_Layer_Factory.register("Image_T", ClippingImageLayer)
Clipping_Layer_Factory.register("NormalMap_T", ClippingNormalMapLayer)