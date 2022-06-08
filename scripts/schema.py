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
class OfmapLayerDimensions:
    width: int
    height: int
    channels: int



@dataclass
class SimResult:
    valid: str
    dram_load: int
    dram_store: int
    weight: int
    psum: int
    ifmap: int
    reuse_chain: int
    pe_util: float
    latency: int
    sim_time: int
    macs: int


@dataclass(frozen=True)
class TestCase:
    ifmap_h: int
    ifmap_w: int
    kernel: int
    c_in: int
    f_out: int
    arch_padding: int
    arch_filter_count: int
    arch_channel_count: int
    groups: int
    lowering_ops: int
    lifting_ops: int
    bias: bool
    layer_name: Optional[str] = None
    model_name: Optional[str] = None

    def get_size(self):
        return self.ifmap_h * self.ifmap_w * self.c_in * self.f_out

    def __lt__(self, other):
        return self.get_size() < other.get_size()
