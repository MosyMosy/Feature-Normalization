from collections import OrderedDict

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from seaborn.palettes import color_palette
import numpy as np

# import seaborn as sns
import torch
import os

import models
from datasets import (Chest_few_shot, CropDisease_few_shot, EuroSAT_few_shot,
                      ISIC_few_shot, miniImageNet_few_shot)


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


def load_checkpoint2(model, load_path, device):
    '''
    Load model and optimizer from load path 
    Return the epoch to continue the checkpoint
    '''
    sd = torch.load(load_path, map_location=torch.device(device))
    model.load_state_dict(sd['model'])

    return sd['epoch']

def get_logs_path(method, target = ''):
    root = './logs'
    method_path = root + '/' + method
    if os.path.isdir(method_path) == False:
        print('The methode {}\'s  path doesn\'t exist'.format(method))
    if target == '':
        return method_path
    log_path = method_path + '/' + target    
    return log_path

def get_feature_list(method, dataloader_list, dataset_names_list):
    model = models.ResNet10()
    load_checkpoint2(
        model, get_logs_path(method) + '/checkpoint_best.pkl', device)
    model.eval()
    label_dataset = []
    feature_list = []
    with torch.no_grad():
        for i, loader in enumerate(dataloader_list):
            # loader_iter = iter(loader)
            # x, _ = loader_iter.next()        
            for x, _ in loader:            
                feature_list += model(x)            
                label_dataset += [dataset_names_list[i]]*len(x)
                # break
    return torch.stack(feature_list).numpy(), label_dataset


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)



dataset_class_list = [miniImageNet_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot, ISIC_few_shot]
dataset_names_list = ['miniImageNet', 'EuroSAT',
                      'CropDisease', 'ChestX', 'ISIC']

dataloader_list = []
for i, dataset_class in enumerate(dataset_class_list):
    transform = dataset_class.TransformLoader(
        224).get_composed_transform(aug=True)
    transform_test = dataset_class.TransformLoader(
        224).get_composed_transform(aug=False)
    # split = 'datasets/split_seed_1/{0}_labeled_20.csv'.format(
    #     dataset_names_list[i])
    # if dataset_names_list[i] == 'miniImageNet':
    split = None
    dataset = dataset_class.SimpleDataset(
        transform, split=split)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256,
                                         num_workers=0,
                                         shuffle=True, drop_last=True)
    dataloader_list.append(loader)

baseline_features, labels = get_feature_list('baseline', dataloader_list, dataset_names_list)
baseline_na_features, labels = get_feature_list('baseline_na', dataloader_list, dataset_names_list)
color = sns.color_palette(n_colors=len(dataloader_list))
perplexitys = range(0, 51, 1)
for perplexity in perplexitys:
    fig = plt.figure(figsize=(20, 10))
    ax = fig.subplots(1,2)    
    
    baseline_embedding = TSNE(perplexity=perplexity, n_iter=5000, learning_rate=10).fit_transform(baseline_features)    
    sns.kdeplot(x=baseline_embedding[:, 0], y=baseline_embedding[:, 1],
                hue=labels, ax=ax[0], palette=color)
    
    baseline_na_embedding = TSNE(perplexity=perplexity, n_iter=5000, learning_rate=10).fit_transform(baseline_na_features)    
    sns.kdeplot(x=baseline_na_embedding[:, 0], y=baseline_na_embedding[:, 1],
                hue=labels, ax=ax[1], palette=color)
    title = 'Left baseline, Right baseline_na. Perplexity {0}'.format(perplexity)
    fig.suptitle(title)
    
    plt.savefig('./lab/tsne/{0}.png'.format(title))
    plt.savefig('./lab/tsne/{0}.svg'.format(title))
    print(title)
