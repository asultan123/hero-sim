from calendar import c
import torch
from collections import OrderedDict, Counter
from functools import partial
from dataclasses import dataclass
from typing import Tuple, List
from copy import deepcopy
from math import prod
from enum import Enum

class LayerType(Enum):
    conv = 1
    linear = 2

@dataclass
class LayerDimensions:
    kernel_size: Tuple[int, int]
    stride: Tuple[int, int]
    padding: Tuple[int, int]
    groups: int
    input_size: List[int]
    output_size: List[int]
    layer_type: LayerType


class ModelStatCollector:
    def __init__(self):
        self.model_stats = OrderedDict()
        self.hooks = []

    def __get_next_conv_layers(self, model):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                yield (name, module)

    def __extract_stats(self, name, module, input, output):
        self.model_stats[name] = LayerDimensions(
            module.kernel_size,
            module.stride,
            module.padding,
            module.groups,
            input_size=list(input[0].size()),
            output_size=list(output[0].size()),
        )

    def __attach_collection_hooks_to_model(self, model):
        for name, conv_layer in self.__get_next_conv_layers(model):
            layer_collector = partial(self.__extract_stats, name)
            self.hooks.append(conv_layer.register_forward_hook(layer_collector))

    def __detach_stats_collection_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def __reset(self):
        self.model_stats = {}
        self.hooks = []

    @classmethod
    def collect_stats_from_model(cls, model, input_batch):
        collector = cls()
        collector.__attach_collection_hooks_to_model(model)
        model.eval()
        with torch.no_grad():
            model(input_batch)
        collector.__detach_stats_collection_hooks()
        collected_stats = deepcopy(collector.model_stats)
        collector.__reset()
        return collected_stats
    
import torch
from PIL import Image
from torchvision import transforms, models
from ModelAnalysis import ModelStatAnalyser
import timm
from tqdm import tqdm
import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from sklearn.preprocessing import MinMaxScaler as Slearn_Scaler
from math import floor, ceil

# prepare sample inputs
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

uris = [
    'http://images.cocodataset.org/val2017/000000397133.jpg'
]
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

inputs = [utils.prepare_input(uri) for uri in uris]
ssd_input_batch = utils.prepare_tensor(inputs)

model_dict = {
    'ssd': torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd'),
    'lenet': torch.hub.load('pytorch/vision:v0.10.0', 'googlenet'),
    'yolov5s': torch.hub.load('ultralytics/yolov5', 'yolov5s'),
    'yolov5m': torch.hub.load('ultralytics/yolov5', 'yolov5m'),
    'yolov5l': torch.hub.load('ultralytics/yolov5', 'yolov5l'),
    'yolov5x': torch.hub.load('ultralytics/yolov5', 'yolov5x'),
    'alexnet': models.alexnet(),
    'squeezenet_1_0': models.squeezenet1_1(),
    'squeezenet_1_1': models.squeezenet1_0(),
    'googlenet': models.googlenet(),
    'shufflenet_v2_x0_5': models.shufflenet_v2_x0_5(),
    'shufflenet_v2_x1_0': models.shufflenet_v2_x1_0(),
}

stats_dict = ModelStatAnalyser.get_models_stats_dict(model_dict, input_batch, ssd_input_batch) 
model_batch_size = 4
model_list = timm.list_models(exclude_filters=['levit_*'], pretrained=False)
in_shapes = []
for model_batch_idx in tqdm(range(0, len(model_list), model_batch_size)):
    for model_name in model_list[model_batch_idx: model_batch_idx+model_batch_size]:
        model = timm.create_model(model_name, pretrained=False)
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)

        url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
        urllib.request.urlretrieve(url, filename)
        img = Image.open(filename).convert('RGB')
        tensor = transform(img).unsqueeze(0) # transform and add batch dimension
        in_tensor_shape = tensor.size()
        if in_tensor_shape[2] == 224 and in_tensor_shape[3] == 224:       
            in_shapes.append(in_tensor_shape)
            model_dict = {}
            model_dict[model_name] = model
            try:
                res_stats_dict = ModelStatAnalyser.get_models_stats_dict(model_dict, tensor, ssd_input_batch)
            except Exception as e:
                print(f"Model name {model_name}")
                raise e
            stats_dict = stats_dict | res_stats_dict
            

import pickle
with open('stats_dict.backup', 'wb') as backup:
    pickle.dump(stats_dict, backup)

