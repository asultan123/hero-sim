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
from schema import IfmapLayerDimensions, LayerDimensions
from joblib import Memory
import os
class StatsCounter:
    def __init__(self):
        self._dict = {}

    def update(self, key, v=None):
        if key not in self._dict:
            self._dict[key] = 1
        else:
            self._dict[key] += v if v is not None else 1

    def __iadd__(self, other):
        if isinstance(other, StatsCounter):
            for k, v in other.items():
                self.update(k, v)
        else:
            raise TypeError("Can only add other StatsCounters to other StatsCounters")
        return self

    def __getitem__(self, key):
        return self._dict[key]

    def __str__(self):
        return self._dict.__str__()

    def __repr__(self):
        return self._dict.__repr__()

    def items(self):
        return self._dict.items()


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
            if input[0].size()[0] != 1:
                raise Exception("Only a batch size of 1 is allowed when extracting model dims")
            
            dims = IfmapLayerDimensions(
                height=prod(list(input[0].size())[:-1]),
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
    def collect_layer_dims_from_model(cls, model, input_batch, use_cuda = False):
        collector = cls()
        collector.attach_collection_hooks_to_model(model)
        if use_cuda and torch.cuda.is_available():
            model = model.cuda()
            input_batch = input_batch.cuda()
        model.eval()
        with torch.no_grad():
            model(input_batch)
        model.cpu()
        input_batch = input_batch.cpu()
        collector.detach_stats_collection_hooks()
        collected_stats = deepcopy(collector.model_stats)
        return collected_stats
    
def load_model_from_timm(model_name):
    return timm.create_model(model_name, pretrained=False)


def load_default_input_tensor_for_model(model):
    url, filename = (
        "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
        "dog.jpg",
    )
    urllib.request.urlretrieve(url, filename)
    img = Image.open(filename).convert("RGB")
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    return transform(img).unsqueeze(0)


if __name__ == "__main__":
    model_name_list = timm.list_models(exclude_filters=['*_iabn', 'swin_*', 'tnt_*', 'tresnet_*'], pretrained=False)
    
    processed_models_list = os.listdir('../data/processed_models')
    processed_models_list = [filename.split('.')[-3] for filename in processed_models_list]
    
    for model_name in tqdm(model_name_list):
        if model_name in processed_models_list:
            continue
        model = load_model_from_timm(model_name)
        input = load_default_input_tensor_for_model(model)
        model_stats = ModelDimCollector.collect_layer_dims_from_model(model, input)
        with open(f'../data/processed_models/{model_name}.model.pickle', 'wb') as model_file:
            pickle.dump(model_stats, model_file)