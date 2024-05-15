"""
This type stub file was generated by pyright.
"""

from dataclasses import dataclass
from typing import List

logger = ...
def list_field(default=..., metadata=...): # -> None:
    ...

@dataclass
class BenchmarkArguments:
    """
    BenchMarkArguments are arguments we use in our benchmark scripts **which relate to the training loop itself**.

    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify them on the command
    line.
    """
    models: List[str] = ...
    batch_sizes: List[int] = ...
    sequence_lengths: List[int] = ...
    inference: bool = ...
    cuda: bool = ...
    tpu: bool = ...
    fp16: bool = ...
    training: bool = ...
    verbose: bool = ...
    speed: bool = ...
    memory: bool = ...
    trace_memory_line_by_line: bool = ...
    save_to_csv: bool = ...
    log_print: bool = ...
    env_print: bool = ...
    multi_process: bool = ...
    inference_time_csv_file: str = ...
    inference_memory_csv_file: str = ...
    train_time_csv_file: str = ...
    train_memory_csv_file: str = ...
    env_info_csv_file: str = ...
    log_filename: str = ...
    repeat: int = ...
    only_pretrain_model: bool = ...
    def __post_init__(self): # -> None:
        ...
    
    def to_json_string(self): # -> str:
        """
        Serializes this instance to a JSON string.
        """
        ...
    
    @property
    def model_names(self) -> List[str]:
        ...
    
    @property
    def do_multi_processing(self): # -> bool:
        ...
    


