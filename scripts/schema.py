from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class LayerDimensions:
    kernel_size: Tuple[int, int]
    stride: Tuple[int, int]
    padding: Tuple[int, int]
    groups: int
    input_size: List[int]
    output_size: List[int]


@dataclass
class IfmapLayerDimensions:
    width: int
    height: int
    channels: int


@dataclass
class SimResult:
    valid: str
    dram: int
    weight: int
    psum: int
    ifmap: int
    pe_util: float
    latency: int
    sim_time: int


@dataclass(frozen=True)
class TestCase:
    ifmap_h: int
    ifmap_w: int
    kernel: int
    c_in: int
    f_out: int
    arch_filter_count: int
    arch_channel_count: int
    layer_name: Optional[str] = None
