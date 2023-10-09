import numpy as np
from .Types import *
from .Parsers import *
import copy

class BaseParserLayer:
    def __init__(self, layer_cfg: Layer_T, sampler_cfg: Parser_T, **kwarg) -> None:
        self._randomizer = None
        self._randomization_space = None

        self._sampler_cfg = copy.copy(sampler_cfg)
        self._layer_cfg = copy.copy(layer_cfg)

    def initializeParser(self) -> None:
        self._sampler = Parser_Factory.get(self._sampler_cfg)

    def sample(self, num: int = 1):
        raise NotImplementedError()

    def __call__(self, num: int = 1) -> np.ndarray([]):
        points = self.sample(num = num)
        return points
    
class DataParserLayer(BaseParserLayer):
    """
    Class which is similar with LineLayer
    But, do not specify bound, since the bound is determined by height image.
    """
    def __init__(self, layer_cfg: Layer_T, sampler_cfg: Sampler_T) -> None:
        super().__init__(layer_cfg, sampler_cfg)
        self.initializeParser()

    def sample(self, num: int = 1):
        return self._sampler(num=num)
    
    def __call__(self, num: int = 1) -> np.ndarray([]):
        points = self.sample(num = num)
        return points

class PerturbatedDataParserLayer(BaseParserLayer):
    """
    Class which is similar with LineLayer
    But, do not specify bound, since the bound is determined by height image.
    """
    def __init__(self, layer_cfg: Layer_T, sampler_cfg: Sampler_T) -> None:
        super().__init__(layer_cfg, sampler_cfg)
        self.initializeParser()

    def sample(self, num: int = 1):
        return self._sampler(num=num)
    
    def __call__(self, num: int = 1) -> np.ndarray([]):
        points = self.sample(num = num)
        return points

class ParserLayerFactory:
    def __init__(self):
        self.creators = {}
    
    def register(self, name: str, class_: BaseParserLayer) -> None:
        self.creators[name] = class_
        
    def get(self, cfg: Layer_T, sampler_cfg: Sampler_T) -> BaseParserLayer:
        if cfg.__class__.__name__ not in self.creators.keys():
            raise ValueError("Unknown layer requested.")
        return self.creators[cfg.__class__.__name__](cfg, sampler_cfg)
    
Parser_Layer_Factory = ParserLayerFactory()
Parser_Layer_Factory.register("Parser_Layer_T", DataParserLayer)
Parser_Layer_Factory.register("Perturbated_Parser_Layer_T", PerturbatedDataParserLayer)