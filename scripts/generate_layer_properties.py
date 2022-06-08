from get_unique_layers_within_models import ConvLayer, LinearLayer
import pickle
import pandas as pd

def calculate_conv_macs(conv_layer: ConvLayer):
    ifmap_w = conv_layer.width
    ifmap_h = conv_layer.height
    channels = conv_layer.in_channels
    filters = conv_layer.out_channels
    padding_h = conv_layer.padding[0]
    padding_w = conv_layer.padding[1]
    kernel_h = conv_layer.kernel_size[0]
    kernel_w = conv_layer.kernel_size[1]
    dilation_h = conv_layer.dilation[0]
    dilation_w = conv_layer.dilation[1]
    groups = conv_layer.groups
    stride_h = conv_layer.stride[0]
    stride_w = conv_layer.stride[1]

    ofmap_h = (
        ((ifmap_h - (kernel_h - 1) * dilation_h + 1 + 2 * padding_h)) / stride_h
    ) + 1
    ofmap_w = (
        ((ifmap_w - (kernel_w - 1) * dilation_w + 1 + 2 * padding_w)) / stride_w
    ) + 1
    macs = ofmap_h * ofmap_w * kernel_h * kernel_w * channels * filters / groups

    return macs

def calculate_conv_ifmap_mem_size(conv_layer: ConvLayer):
    ifmap_w = conv_layer.width
    ifmap_h = conv_layer.height
    channels = conv_layer.in_channels
    return ifmap_h * ifmap_w * channels


def calculate_conv_mem_ofmap_mem_size(conv_layer: ConvLayer):
    ifmap_w = conv_layer.width
    ifmap_h = conv_layer.height
    filters = conv_layer.out_channels
    padding_h = conv_layer.padding[0]
    padding_w = conv_layer.padding[1]
    kernel_h = conv_layer.kernel_size[0]
    kernel_w = conv_layer.kernel_size[1]
    dilation_h = conv_layer.dilation[0]
    dilation_w = conv_layer.dilation[1]
    stride_h = conv_layer.stride[0]
    stride_w = conv_layer.stride[1]
    
    ofmap_h = (
        ((ifmap_h - (kernel_h - 1) * dilation_h + 1 + 2 * padding_h)) / stride_h
    ) + 1
    ofmap_w = (
        ((ifmap_w - (kernel_w - 1) * dilation_w + 1 + 2 * padding_w)) / stride_w
    ) + 1
    
    return ofmap_h*ofmap_w*filters

def calculate_linear_macs(linear_layer: LinearLayer):
    in_features = linear_layer.in_features
    out_features = linear_layer.out_features
    channels = linear_layer.channels
    height = linear_layer.height
    width = linear_layer.width
    assert channels == in_features
    return (height*width)*in_features*out_features


def calculate_linear_ifmap_mem_size(linear_layer: LinearLayer):
    in_features = linear_layer.in_features
    channels = linear_layer.channels
    height = linear_layer.height
    width = linear_layer.width
    assert channels == in_features
    return (height*width)*in_features


def calculate_linear_ofmap_mem_size(linear_layer: LinearLayer):
    in_features = linear_layer.in_features
    out_features = linear_layer.out_features
    channels = linear_layer.channels
    assert channels == in_features
    return in_features*out_features

with open("../data/model_unique_layers_tracker.pickle", "rb") as file:
    model_unique_layers_tracker = pickle.load(file)


model_dict = {}
for model, layer_dict in model_unique_layers_tracker.items():
    layer_property_dict = {} 
    for layer, layer_names_list in layer_dict.items():
        if isinstance(layer, ConvLayer):
            properties = {
                "macs": calculate_conv_macs(layer),
                "ifmap_mem_size": calculate_conv_ifmap_mem_size(layer),
                "ofmap_mem_size": calculate_conv_mem_ofmap_mem_size(layer)
            }
        elif isinstance(layer, LinearLayer):
            properties = {
                "macs": calculate_linear_macs(layer),
                "ifmap_mem_size": calculate_linear_ifmap_mem_size(layer),
                "ofmap_mem_size": calculate_linear_ofmap_mem_size(layer)
            }
        else:
            raise Exception(f"Invalid layer type {type(layer)}")
        for layer_name in layer_names_list:
            layer_property_dict[layer_name] = properties
    for layer_name, properties in layer_property_dict.items():
        model_dict[(model, layer_name)] = properties
        
pd.DataFrame.from_dict(model_dict, orient='index').to_csv('../data/layer_properties.csv')