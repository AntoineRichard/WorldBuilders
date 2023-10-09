import numpy as np
from .Types import *

## Parse pickle dumped data into numpy ndarray

class BaseParser:
    def __init__(self, parser_cfg:Parser_T):
        self.data_parsed = parser_cfg.data
    def __call__(self, **kwargs):
        return self.sample(**kwargs)
    def sample(self, **kwargs):
        raise NotImplementedError

class DataParser:
    def __init__(self, parser_cfg:Parser_T):
        self.data_parsed = parser_cfg.data
    def __call__(self, **kwargs):
        return self.sample(**kwargs)
    def sample(self, **kwargs):
        return self.data_parsed

class PerturbatedDataParser(BaseParser):
    """
    Add gaussian perturbation to the original data parsed
    """
    def __init__(self, parser_cfg:Parser_T):
        super().__init__(parser_cfg)
        self._parser_cfg = parser_cfg
        if self._parser_cfg.seed != -1:
            self._rng = np.random.default_rng(self._parser_cfg.seed)
        else:
            self._rng = np.random.default_rng()
    def sample(self, **kwargs):
        noise = self._rng.normal(self._parser_cfg.mean, self._parser_cfg.std, self.data_parsed.shape)
        return self.data_parsed * (1+noise)

class ParserFactory:
    def __init__(self):
        self.creators = {}
    
    def register(self, name: str, class_: BaseParser) -> None:
        self.creators[name] = class_
        
    def get(self, cfg: Sampler_T, **kwargs:dict) -> BaseParser:
        if cfg.__class__.__name__ not in self.creators.keys():
            raise ValueError("Unknown sampler requested.")
        return self.creators[cfg.__class__.__name__](cfg)

Parser_Factory = ParserFactory()
Parser_Factory.register("Parser_T", DataParser)
Parser_Factory.register("Perturbated_Parser_T", PerturbatedDataParser)