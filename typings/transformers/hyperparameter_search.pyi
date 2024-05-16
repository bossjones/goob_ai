"""
This type stub file was generated by pyright.
"""

logger = ...
class HyperParamSearchBackendBase:
    name: str
    pip_package: str = ...
    @staticmethod
    def is_available():
        ...
    
    def run(self, trainer, n_trials: int, direction: str, **kwargs):
        ...
    
    def default_hp_space(self, trial):
        ...
    
    def ensure_available(self): # -> None:
        ...
    
    @classmethod
    def pip_install(cls): # -> str:
        ...
    


class OptunaBackend(HyperParamSearchBackendBase):
    name = ...
    @staticmethod
    def is_available(): # -> bool:
        ...
    
    def run(self, trainer, n_trials: int, direction: str, **kwargs): # -> BestRun:
        ...
    
    def default_hp_space(self, trial): # -> Dict[str, float]:
        ...
    


class RayTuneBackend(HyperParamSearchBackendBase):
    name = ...
    pip_package = ...
    @staticmethod
    def is_available(): # -> bool:
        ...
    
    def run(self, trainer, n_trials: int, direction: str, **kwargs): # -> BestRun:
        ...
    
    def default_hp_space(self, trial): # -> Dict[str, float]:
        ...
    


class SigOptBackend(HyperParamSearchBackendBase):
    name = ...
    @staticmethod
    def is_available(): # -> bool:
        ...
    
    def run(self, trainer, n_trials: int, direction: str, **kwargs): # -> BestRun:
        ...
    
    def default_hp_space(self, trial): # -> list[dict[str, Any]]:
        ...
    


class WandbBackend(HyperParamSearchBackendBase):
    name = ...
    @staticmethod
    def is_available(): # -> bool:
        ...
    
    def run(self, trainer, n_trials: int, direction: str, **kwargs): # -> BestRun:
        ...
    
    def default_hp_space(self, trial): # -> Dict[str, float]:
        ...
    


ALL_HYPERPARAMETER_SEARCH_BACKENDS = ...
def default_hp_search_backend() -> str:
    ...

