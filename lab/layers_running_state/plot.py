from collections import OrderedDict
from math import trunc
import math
import statistics
import joypy
import matplotlib
import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
from pandas.core import groupby
import torch
import torch.nn as nn
from datasets import (Chest_few_shot, CropDisease_few_shot, EuroSAT_few_shot,
                      ISIC_few_shot, miniImageNet_few_shot)
from lab.layers.resnet10 import ResNet10
from lab.layers.resnet import resnet18
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from scipy.stats.stats import mode

from PIL import Image


def load_checkpoint(model, load_path, device):
    '''
    Load model and optimizer from load path 
    Return the epoch to continue the checkpoint
    '''
    state = torch.load(load_path, map_location=torch.device(device))['state']
    clf_state = OrderedDict()
    state_keys = list(state.keys())
    for _, key in enumerate(state_keys):
        if "feature." in key:
            # an architecture model has attribute 'feature', load architecture
            # feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
            newkey = key.replace("feature.", "")
            state[newkey] = state.pop(key)
        elif "classifier." in key:
            newkey = key.replace("classifier.", "")
            clf_state[newkey] = state.pop(key)
        else:
            state.pop(key)
    model.load_state_dict(state)
    model.eval()
    return model


def load_checkpoint2(model, load_path, device):
    '''
    Load model and optimizer from load path 
    Return the epoch to continue the checkpoint
    '''
    sd = torch.load(load_path, map_location=torch.device(device))
    model.load_state_dict(sd['model'])
    model.eval()
    return model

def load_checkpoint3(model, load_path, device):
    '''
    Load model and optimizer from load path 
    Return the epoch to continue the checkpoint
    '''
    sd = torch.load(load_path, map_location=torch.device(device))
    # model = nn.DataParallel(model)
        
    model.load_state_dict(sd['state_dict'])
    model.eval()
    return model


def get_BN_output(model, colors, layers=None, channels=None, position='before_affine', flatten = False):
    newcolors = []
    labels = []
    BN_list = []
    if (layers is None) or flatten:
        flatten = True
    else:
        flatten = False

    
    i = 0
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            if (layers is None) or (i in layers):
                flat_list = []
                out = {'input': layer.input.clone(), 'output': layer.output.clone()}.get(
                    position, layer.before_affine.clone())                
                out = out.permute(1,0,2,3)
                out = out.flatten(start_dim=1).squeeze().tolist()
                for j, channel in enumerate(out):
                    if (channels is None) or (j in channels):
                        if flatten:
                            flat_list += channel
                        else:
                            flat_list.append(channel)

                if flatten:
                    BN_list.append(flat_list)
                    labels += ['Layer {0:02d} ({1: 0.2f}, {2: 0.2f})'.format(
                        i+1, torch.tensor(flat_list).mean(),torch.tensor(flat_list).std())]
                else:
                    BN_list += flat_list
                    if (channels is not None) and (len(channels) == 1):
                        labels += ['Layer {0:02d} ({1: 0.2f}, {2: 0.2f})'.format(
                            i+1, statistics.mean(flat_list[0]), statistics.stdev(flat_list[0]))]
                    else:
                        labels += ['Layer {0:02d}'.format(i+1)]
                    if channels is None:
                        labels += [None]*(len(out)-1)
                    else:
                        labels += [None]*(len(channels)-1)

                    clm = LinearSegmentedColormap.from_list(
                        "Custom", colors, N=len(out))
                    temp = clm(range(0, len(out)))
                    for c in temp:
                        newcolors.append(c)

            i += 1
    if flatten:
        clm = LinearSegmentedColormap.from_list(
            "Custom", colors, N=i)
        temp = clm(range(0, i))
        for c in temp:
            newcolors.append(c)

    return BN_list, labels, ListedColormap(newcolors, name='custom')


def to_grid(path_list, out="lab/layers/grid.png", shape=(4, 2)):
    blank_image = None
    for i in range(shape[0]):
        for j in range(shape[1]):
            current_index = i * shape[1] + j
            if current_index > (len(path_list) - 1):
                continue
            image = Image.open(path_list[current_index], mode='r')
            if blank_image is None:
                blank_image = Image.new(mode='RGB', size=(
                    image.width * shape[1], image.height * shape[0]))
            blank_image.paste(image, (j * image.width, i * image.height))
    blank_image.save(out)


def compare_domains(models, base_x, EuroSAT_x, color_range, layers=[[None]], channels=None, value_position='before_affine'):
    for l in layers:
        path_list = []
        for i, model in enumerate(models):
            with torch.no_grad():
                model(base_x)
                mini_out, mini_labels, clm = get_BN_output(
                    model, colors=color_range[i], layers=l, channels=channels, position=value_position)

                model(EuroSAT_x)
                Euro_out, EuroSAT_labels, clm = get_BN_output(
                    model, colors=color_range[i], layers=l, channels=channels, position=value_position)

                args = {'overlap': 4, 'bw_method': 0.2,
                        'colormap': clm, 'linewidth': 0.3, 'x_range': [-2, 2], 'linecolor': 'w',
                        'background': 'w',  'alpha': 0.8, 'figsize': (10, 5), 'fill': True,
                        'grid': False, 'kind': 'kde', 'hist': False, 'bins': int(len(base_x))}

                joypy.joyplot(list(reversed(mini_out)), labels=list(
                    reversed(mini_labels)), **args)
                # plt.show()
                path_list.append(
                    "./lab/layers/{0}_to_MiniImageNet{1}.png".format(model_names[i], '_' + value_position))
                plt.savefig(path_list[-1],)

                joypy.joyplot(list(reversed(Euro_out)), labels=list(
                    reversed(EuroSAT_labels)), **args)
                # plt.show()
                path_list.append(
                    "./lab/layers/{0}_to_EuroSAT{1}.png".format(model_names[i], '_' + value_position))
                plt.savefig(path_list[-1],)
        to_grid(
            path_list, out="lab/layers/compare_domains_{0}{1}.png".format(l,  '_' + value_position), shape=(len(models), 2))


def compare_positions(models, data_x, color_range, layers=[[None]], channels=None):
    for l in layers:
        path_list = []
        for i, model in enumerate(models):
            with torch.no_grad():
                model(data_x)
                input_out, input_labels, clm = get_BN_output(
                    model, colors=color_range[i], layers=l, channels=channels, position='input')
                b_affine_out, b_affine_labels, clm = get_BN_output(
                    model, colors=color_range[i], layers=l, channels=channels, position='before_affine')
                output_out, output_labels, clm = get_BN_output(
                    model, colors=color_range[i], layers=l, channels=channels, position='output')

                args = {'overlap': 4, 'bw_method': 0.2,
                        'colormap': clm, 'linewidth': 0.3, 'linecolor': 'w',
                        'background': 'w',  'alpha': 0.8, 'figsize': (10, 5), 'fill': True,
                        'grid': False, 'kind': 'kde', 'hist': False, 'bins': int(len(data_x))}

                joypy.joyplot(list(reversed(input_out)), labels=list(
                    reversed(input_labels)), **args)
                path_list.append(
                    "./lab/layers/{0}_BN's_{1}.png".format(model_names[i], 'input'))
                plt.savefig(path_list[-1],)

                joypy.joyplot(list(reversed(b_affine_out)), labels=list(
                    reversed(b_affine_labels)), **args)
                path_list.append(
                    "./lab/layers/{0}_BN's{1}.png".format(model_names[i], 'before_affine'))
                plt.savefig(path_list[-1],)

                joypy.joyplot(list(reversed(output_out)), labels=list(
                    reversed(output_labels)), **args)
                path_list.append(
                    "./lab/layers/{0}_BN's_{1}.png".format(model_names[i], 'output'))
                plt.savefig(path_list[-1],)

        to_grid(
            path_list, out="lab/layers/compare_positions_{0}_{1}.png".format(l,  'positions'), shape=(len(models), 3))

def compare_models(models, data_x, color_range, layers=[[None]], channels=None, value_position='input'):
    # for l in layers:
    path_list = []
    for i, model in enumerate(models):
        with torch.no_grad():
            model(data_x)
            out, labels, clm = get_BN_output(
                model, colors=color_range[math.floor(i/2)], layers=layers, channels=channels, position=value_position)

            args = {'overlap': 4, 'bw_method': 0.2,
                    'colormap': clm, 'linewidth': 0.3, 'linecolor': 'w',
                    'background': 'w',  'alpha': 0.8, 'figsize': (10, 5), 'fill': True,
                    'grid': False, 'kind': 'kde', 'hist': False, 'bins': int(len(base_x))}

            joypy.joyplot(list(reversed(out)), labels=list(
                reversed(labels)), **args)
            # plt.show()
            path_list.append(
                "./lab/layers/{0}.png".format(i))
            plt.savefig(path_list[-1],)
            break
        # to_grid(
        #     path_list, out="lab/layers/compare_models_{0}{1}.png".format(l,  '_' + value_position), shape=(int(len(models)/2), 2))

def compare_two_models(model, model_na, data_x, color_range, layers=None, channels=None, value_position='input'):
    df = pd.DataFrame()    
    path_list = []      
    with torch.no_grad():
        model(data_x)
        model_na(data_x)
        out, labels, clm = get_BN_output(
            model, colors=color_range[0], layers=layers, channels=channels, position=value_position, flatten=True)
        out_na, labels, clm = get_BN_output(
            model_na, colors=color_range[0], layers=layers, channels=channels, position=value_position, flatten=True)

        args = {'overlap': 4, 'bw_method': 0.2,
                'linewidth': 1, 'legend':True, 'color':["#00E7E7","#008181"],
                'background': 'w',  'alpha': 0.5, 'figsize': (10, 15), 'fill': True, 'x_range':[-5,5],
                'grid': False, 'kind': 'kde', 'hist': False, 'bins': int(len(base_x))}
    if layers is None: layers = range(len(out))
    out_list = []
    out_list_na = []
    layer_list = []
    for i, (l, l_na) in enumerate(zip(out, out_na)):
        out_list += l
        out_list_na += l_na
        layer_list += ['layer {0:02d}'.format(layers[i])] * len(l)
    df['with affine'] = out_list
    df['without affine'] = out_list_na
    df['layer'] = layer_list
    print('ploting')
    joypy.joyplot(df, by='layer', **args)
    # 
    path_list.append(
        "./lab/layers/compare_two_models_{0}.pdf".format(value_position))

    plt.savefig(path_list[-1],)
    # plt.show()


device = torch.device("cpu")
models = []
models_na = []

model_names_na = ['Baseline_na', 'BMS_in_Eurosat_na']
models_na.append(load_checkpoint3(
    resnet18(), 'logs/ImageNet_na/checkpoint_best.pkl', device))
models_na.append(load_checkpoint2(
    ResNet10(), 'logs/BMS_in_na/EuroSAT/checkpoint_best.pkl', device))

model_names = ['Baseline', 'BMS_in_Eurosat',
               'AdaBN_EuroSAT', 'STARTUP_EuroSAT']
models.append(load_checkpoint3(
    resnet18(), 'logs/ImageNet/checkpoint_best.pkl', device))
models.append(load_checkpoint2(
    ResNet10(), 'logs/BMS_in/EuroSAT/checkpoint_best.pkl', device))
models.append(load_checkpoint2(
    ResNet10(), 'logs/AdaBN/EuroSAT/checkpoint_best.pkl', device))
models.append(load_checkpoint2(
    ResNet10(), 'logs/STARTUP/EuroSAT/checkpoint_best.pkl', device))


b_size = 64
transform = EuroSAT_few_shot.TransformLoader(
    224).get_composed_transform(aug=True)
transform_test = EuroSAT_few_shot.TransformLoader(
    224).get_composed_transform(aug=False)
split = 'datasets/split_seed_1/EuroSAT_unlabeled_20.csv'
dataset = EuroSAT_few_shot.SimpleDataset(
    transform, split=split)
EuroSAT_loader = torch.utils.data.DataLoader(dataset, batch_size=b_size,
                                             num_workers=0,
                                             shuffle=True, drop_last=True)

transform = miniImageNet_few_shot.TransformLoader(
    224).get_composed_transform(aug=True)
transform_test = miniImageNet_few_shot.TransformLoader(
    224).get_composed_transform(aug=False)
dataset = miniImageNet_few_shot.SimpleDataset(
    transform, split=None)
base_loader = torch.utils.data.DataLoader(dataset, batch_size=b_size,
                                          num_workers=0,
                                          shuffle=True, drop_last=True)

EuroSAT_x, _ = iter(EuroSAT_loader).next()
base_x, _ = iter(base_loader).next()


color_range = [['#670022', '#FF6699'], ['#004668', '#66D2FF'],
               ['#9B2802', '#FF9966'], ['#346600', '#75E600']]

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
layers = [[i] for i in range(12)]# [None] is for full network
channels = None

# compare_domains(models=models, base_x=base_x, EuroSAT_x=EuroSAT_x, color_range=color_range,
#                 layers=layers, channels=channels, value_position='input')

# compare_positions(models, data_x=base_x,
#                   color_range=color_range, layers=layers, channels=channels)

# compare_models([models[0]], data_x=base_x, color_range=color_range, layers=None, channels=channels,value_position='output')

# layers = range(2,5)
compare_two_models(models[0], models_na[0], data_x=EuroSAT_x, color_range=color_range, layers=None, channels=channels,value_position='input')
compare_two_models(models[0], models_na[0], data_x=EuroSAT_x, color_range=color_range, layers=None, channels=channels,value_position='output')
compare_two_models(models[0], models_na[0], data_x=EuroSAT_x, color_range=color_range, layers=None, channels=channels,value_position='before_affine')