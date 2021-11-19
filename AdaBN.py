import random
import math
import copy
from datasets import miniImageNet_few_shot, tiered_ImageNet_few_shot, ImageNet_few_shot
from datasets import ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot
from collections import OrderedDict
import warnings
import models
import time
import data
import utils
import sys
import numpy as np
import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms, datasets
import torch.utils.data
from configs import miniImageNet_path, ISIC_path, ChestX_path, CropDisease_path, EuroSAT_path

torch.cuda.empty_cache()


sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


# import wandb


class apply_twice:
    '''
        A wrapper for torchvision transform. The transform is applied twice for 
        SimCLR training
    '''

    def __init__(self, transform, transform2=None):
        self.transform = transform

        if transform2 is not None:
            self.transform2 = transform2
        else:
            self.transform2 = transform

    def __call__(self, img):
        return self.transform(img), self.transform2(img)


def main(args):
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    torch.cuda.empty_cache()
    # Set the scenes
    if not os.path.isdir(args.dir):
        os.makedirs(args.dir)

    logger = utils.create_logger(os.path.join(
        args.dir, time.strftime("%Y%m%d-%H%M%S") + '_checkpoint.log'), __name__)
    trainlog = utils.savelog(args.dir, 'train')
    vallog = utils.savelog(args.dir, 'val')

    # wandb.init(project='STARTUP',
    #            group=__file__,
    #            name=f'{__file__}_{args.dir}')

    # wandb.config.update(args)

    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    # seed the random number generator
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    ###########################
    # Create Models
    ###########################
    if args.model == 'resnet10':
        backbone = models.ResNet10()

    ############################

    ###########################
    # Create DataLoader
    ###########################

    # create the target dataset
    if args.target_dataset == 'ISIC':
        transform = ISIC_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=True)
        transform_test = ISIC_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=False)
        dataset = ISIC_few_shot.SimpleDataset(
            transform, split=args.target_subset_split)
    elif args.target_dataset == 'EuroSAT':
        transform = EuroSAT_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=True)
        transform_test = EuroSAT_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=False)
        dataset = EuroSAT_few_shot.SimpleDataset(
            transform, split=args.target_subset_split)
    elif args.target_dataset == 'CropDisease':
        transform = CropDisease_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=True)
        transform_test = CropDisease_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=False)
        dataset = CropDisease_few_shot.SimpleDataset(
            transform, split=args.target_subset_split)
    elif args.target_dataset == 'ChestX':
        transform = Chest_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=True)
        transform_test = Chest_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=False)
        dataset = Chest_few_shot.SimpleDataset(
            transform, split=args.target_subset_split)
    elif args.target_dataset == 'miniImageNet_test':
        transform = miniImageNet_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=True)
        transform_test = miniImageNet_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=False)
        dataset = miniImageNet_few_shot.SimpleDataset(
            transform, split=args.target_subset_split)
    elif args.target_dataset == 'ImageNet_test':
        transform = ImageNet_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=True)
        transform_test = ImageNet_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=False)
        dataset = ImageNet_few_shot.SimpleDataset(
            transform, split=args.target_subset_split)
    elif args.target_dataset == 'tiered_ImageNet_test':
        if args.image_size != 84:
            warnings.warn("Tiered ImageNet: The image size for is not 84x84")
        transform = tiered_ImageNet_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=True)
        transform_test = tiered_ImageNet_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=False)
        dataset = tiered_ImageNet_few_shot.SimpleDataset(
            transform, split=args.target_subset_split)
    else:
        raise ValueError('Invalid dataset!')

    print("Size of target dataset", len(dataset))
    dataset_test = copy.deepcopy(dataset)

    transform_twice = apply_twice(transform)
    transform_test_twice = apply_twice(transform_test, transform)

    dataset.d.transform = transform_twice
    dataset_test.d.transform = transform_test_twice

    ind = torch.randperm(len(dataset))

    # split the target dataset into train and val
    # 10% of the unlabeled data is used for validation
    train_ind = ind[:int(0.9*len(ind))]
    val_ind = ind[int(0.9*len(ind)):]

    # trainset = torch.utils.data.Subset(dataset, train_ind)
    # valset = torch.utils.data.Subset(dataset_test, val_ind)
    trainset = dataset

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bsize,
                                              num_workers=args.num_workers,
                                              shuffle=True, drop_last=True)

    #######################################
    starting_epoch = 0

    logger.info('load the pretrained model')
    load_checkpoint(
        backbone, args.base_dictionary, device)

    for epoch in range(starting_epoch, args.epochs):
        addapt(backbone, trainloader, epoch, args.epochs, logger, args, device)

    checkpoint(backbone, os.path.join(
        args.dir,  f'checkpoint_best.pkl'), args.epochs)


def checkpoint(model, save_path, epoch):
    '''
    epoch: the number of epochs of training that has been done
    Should resume from epoch
    '''
    sd = {
        'model': copy.deepcopy(model.state_dict()),
        'epoch': epoch
    }

    torch.save(sd, save_path)
    return sd


def load_checkpoint(model, load_path, device):
    '''
    Load model and optimizer from load path 
    Return the epoch to continue the checkpoint
    '''
    sd = torch.load(load_path, map_location=torch.device(device))
    model.load_state_dict(sd['model'])
    model.eval()


def addapt(model, trainloader, epoch,
           num_epochs, logger, args, device, turn_off_sync=False):

    model.to(device)
    model.train()

    end = time.time()
    for i, ((X1, X2), _) in enumerate(trainloader):
        with torch.no_grad():
            X1 = X1.to(device)
            X2 = X2.to(device)

            #  shift the affine
            f1 = model(X1)
            f2 = model(X2)

            # for layer in model.modules():
            #     if isinstance(layer, nn.BatchNorm2d):
            #         print(layer.running_mean[2])
            #         print(layer.running_var[0])
            #         break
            if (i + 1) % args.print_freq == 0:
                logger_string = ('Training Epoch: [{epoch}/{epochs}] Step: [{step} / {steps}]'
                                 ).format(
                    epoch=epoch, epochs=num_epochs, step=i+1, steps=len(trainloader))

                logger.info(logger_string)
                print(logger_string)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='STARTUP')
    parser.add_argument('--dir', type=str, default='./logs/AdaBN/EuroSAT',
                        help='directory to save the checkpoints')

    parser.add_argument('--bsize', type=int, default=32,
                        help='batch_size for STARTUP')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='Frequency (in epoch) to save')
    parser.add_argument('--eval_freq', type=int, default=2,
                        help='Frequency (in epoch) to evaluate on the val set')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='Frequency (in step per epoch) to print training stats')
    parser.add_argument('--load_path', type=str, default=None,
                        help='Path to the checkpoint to be loaded')
    parser.add_argument('--seed', type=int, default=1,
                        help='Seed for randomness')
    parser.add_argument('--wd', type=float, default=1e-4,
                        help='Weight decay for the model')
    parser.add_argument('--resume_latest', action='store_true',
                        help='resume from the latest model in args.dir')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for dataloader')

    parser.add_argument('--iteration_bp', type=int,
                        help='which step to break in the training loop')
    parser.add_argument('--model', type=str, default='resnet10',
                        help='Backbone model')

    parser.add_argument('--backbone_random_init', action='store_true',
                        help="Use random initialized backbone ")

    parser.add_argument('--base_dataset', type=str,
                        default='miniImageNet', help='base_dataset to use')
    parser.add_argument('--base_path', type=str,
                        default=miniImageNet_path, help='path to base dataset')
    parser.add_argument('--base_split', type=str,
                        help='split for the base dataset')
    parser.add_argument('--base_no_color_jitter', action='store_true',
                        help='remove color jitter for ImageNet')
    parser.add_argument('--base_val_ratio', type=float, default=0.05,
                        help='amount of base dataset set aside for validation')

    parser.add_argument('--batch_validate', action='store_true',
                        help='to do batch validate rather than validate on the full dataset (Ideally, for SimCLR,' +
                        ' the validation should be on the full dataset but might not be feasible due to hardware constraints')

    parser.add_argument('--target_dataset', type=str, default='EuroSAT',
                        help='the target domain dataset')
    parser.add_argument('--target_subset_split', type=str, default='datasets/split_seed_1/EuroSAT_unlabeled_20.csv',
                        help='path to the csv files that specifies the unlabeled split for the target dataset')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Resolution of the input image')
    parser.add_argument('--base_dictionary',  help='base model to addapt')
    args = parser.parse_args()
    main(args)
