"""
This type stub file was generated by pyright.
"""

import tensorflow as tf
from ..configuration_utils import PretrainedConfig
from ..utils import is_py3nvml_available, is_tf_available
from .benchmark_utils import Benchmark
from .benchmark_args_tf import TensorFlowBenchmarkArguments

"""
    Benchmarking the library on inference and training in PyTorch.
"""
if is_tf_available():
    ...
if is_py3nvml_available():
    ...
logger = ...
def run_with_tf_optimizations(do_eager_mode: bool, use_xla: bool): # -> Callable[..., _Wrapped[Callable[..., Any], Any, Callable[..., Any], Any]]:
    ...

def random_input_ids(batch_size: int, sequence_length: int, vocab_size: int) -> [tf.Tensor]:
    ...

class TensorFlowBenchmark(Benchmark):
    args: TensorFlowBenchmarkArguments
    configs: PretrainedConfig
    framework: str = ...
    @property
    def framework_version(self):
        ...
    


