from argparse import ArgumentError
from calendar import c
from optparse import Option
from click import option
import torch
from collections import OrderedDict, Counter
from functools import partial
from dataclasses import dataclass
from typing import Tuple, List
from copy import deepcopy
from math import prod
from enum import Enum
import torch
from PIL import Image
from torchvision import transforms, models
from tqdm import tqdm
import urllib
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from math import floor, ceil
import numpy as np
import urllib
import pickle
from typing import Optional
from torch.nn.modules.linear import Linear
from torch.nn.modules.conv import Conv2d

@dataclass
class IfmapLayerDimensions:
    width: int
    height: int
    channels: int

class ModelDimCollector:
    default_targets = [Conv2d, Linear]

    def __init__(self):
        self.model_stats = OrderedDict()
        self.hooks = []

    def get_next_target_layer(
        self,
        model,
        target_layers=default_targets,
    ):
        for name, module in model.named_modules():
            if type(module) in target_layers:
                yield (name, module)

    def extract_dims(self, name, module, input, output):
        if isinstance(module, Linear):
            dims = IfmapLayerDimensions(
                height=1,
                width=1,
                channels=input[0].size()[-1],
            )
            
        elif isinstance(module, Conv2d):
            dims = IfmapLayerDimensions(
                channels=input[0].size()[-3],
                height=input[0].size()[-2],
                width=input[0].size()[-1],
            )
            
        else:
            raise TypeError(
                f"Unsupported module type {type(module)} found during dimensions extraction"
            )
        self.model_stats[name] = (dims, module)

    def attach_collection_hooks_to_model(self, model):
        for name, layer in self.get_next_target_layer(model):
            layer_collector = partial(self.extract_dims, name)
            self.hooks.append(layer.register_forward_hook(layer_collector))

    def detach_stats_collection_hooks(self):
        for hook in self.hooks:
            hook.remove()

    @classmethod
    def collect_layer_dims_from_model(cls, model, input_batch):
        collector = cls()
        collector.attach_collection_hooks_to_model(model)
        if torch.cuda.is_available():
            model.cuda()
            input_batch.cuda()
        model.eval()
        with torch.no_grad():
            model(input_batch)
        model.cpu()
        collector.detach_stats_collection_hooks()
        collected_stats = deepcopy(collector.model_stats)
        return collected_stats


model_list = [
    (model_name, timm.create_model(model_name, pretrained=False))
    for model_name in ["resnet50", "vgg19", "mobilenetv3_rw"]
]
model_input_image_config = {
    model_name: resolve_data_config({}, model=model) for model_name, model in model_list
}

last_config = list(model_input_image_config.values())[0]
for config in list(model_input_image_config.values())[1:]:
    if config["input_size"] != last_config["input_size"]:
        raise Exception(
            "Mismatch in required input image dimensions for desired models"
        )

url, filename = (
    "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
    "dog.jpg",
)
urllib.request.urlretrieve(url, filename)
img = Image.open(filename).convert("RGB")

stats_dict = {}
for model_name, model in model_list:
    transform = create_transform(**model_input_image_config[model_name])
    input_batch = transform(img).unsqueeze(0)
    stats_dict[model_name] = ModelDimCollector.collect_layer_dims_from_model(
        model, input_batch
    )
print('...')