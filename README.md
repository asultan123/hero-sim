# Description
A cycle accurate simulation framework for the novel AI inference architecture HERO (A Hybrid GEMM and Direct Convolution Accelerator). A full explanation of Hero's internals is available [here](https://github.com/asultan123/MSPaper/blob/master/thesis.pdf) (Document still WIP). The backend simulation enviornment is built using SystemC. This simulation framework can be used to estimate the performance of any arbitrary pytorch model (provided that the model has supported layers) on a range of possible configurations for HERO. 

# Installation
1. Clone this repo
2. Include frontend.py script/ (Migration to pypi in progress)
3. Download latest backend release [here](https://github.com/asultan123/hero-sim/releases)
4. Install backend linux deb package

# Usage
```python
config = {
    "filter_count": 32,
    "channel_count": 18,
    "directly_supported_kernels": [(1, 1), (3, 3)],
    "ifmap_mem_ub": 2**20 // 18 * 18,
    "allow_ifmap_distribution": True,
    "ofmap_mem_ub": 2**20,
    "allow_ofmap_distribution": True,
    "reuse_chain_bank_size": 512,
    "weight_bank_size": 16,
    "groups_supported": False,
}

from frontend import eval_network
result_dataframe = eval_network(model=pytorch_model, arch_config=config)
```

# Limitations
1. Layer types are limited to Linear and Conv2D
2. DRAM bandwidth restrictions are not considered in latency estimations

# Simulation Architecture Overview
## Frontend
![](https://i.imgur.com/yccY4iT.png)
## Backend
![](https://i.imgur.com/pazJCo4.png)

# Hero Architecture Overview
![](https://i.imgur.com/jOmoitJ.png)
