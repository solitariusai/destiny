from os import PathLike
from pathlib import Path


def _save_safetensors(
    state_dict: dict, 
    path: str |PathLike, 
    filename: str, 
    extension: str, 
    max_shard_byte_size: int
) -> list[Path]: 
    ...


def sum_as_string(a: int, b: int) -> str: ...

