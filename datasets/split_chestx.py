# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from abc import abstractmethod

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from shutil import copyfile
import sys
sys.path.append("../")


# identity = lambda x:x
def identity(x): return x

class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path="Data_Entry_2017.csv",
        image_path = "/images/", split="ChestX_unlabeled_20.csv"):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
            target_transform: pytorch transforms for targets
            split: the filename of a csv containing a split for the data to be used. 
                    If None, then the full dataset is used. (Default: None)
        """
        self.img_path = image_path
        self.csv_path = csv_path
        self.used_labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax"]

        self.labels_maps = {"Atelectasis": 0, "Cardiomegaly": 1, "Effusion": 2, "Infiltration": 3, "Mass": 4, "Nodule": 5,  "Pneumothorax": 6}
        
        labels_set = []

        
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, skiprows=[0], header=None)

        # First column contains the image paths
        self.image_name_all = np.asarray(self.data_info.iloc[:, 0])
        self.labels_all = np.asarray(self.data_info.iloc[:, 1])

        self.image_name  = []
        self.labels = []

        self.split = split


        for name, label in zip(self.image_name_all,self.labels_all):
            label = label.split("|")

            if len(label) == 1 and label[0] != "No Finding" and label[0] != "Pneumonia" and label[0] in self.used_labels:
                self.labels.append(self.labels_maps[label[0]])
                self.image_name.append(name)
    
        self.data_len = len(self.image_name)
        self.image_name = np.asarray(self.image_name)
        self.labels = np.asarray(self.labels)

        if split is not None:
            print("Using Split: ", split)
            split = pd.read_csv(split)['img_path'].values
            # construct the index
            ind = np.concatenate(
                [np.where(self.image_name == j)[0] for j in split])
            self.image_name = self.image_name[ind]
            self.labels = self.labels[ind]
            self.data_len = len(split)

            assert len(self.image_name) == len(split)
            assert len(self.labels) == len(split)
        # self.targets = self.labels

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_name[index]

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.labels[index]

        return single_image_name, single_image_label

    def __len__(self):
        return self.data_len


ds = CustomDatasetFromImages()

with open('result.txt', 'w') as sample_file:
    for i in range(len(ds)):
        sample = ds[i]
        sample_file.write('{0},{1}\n'.format(sample[0], sample[1]))
        copyfile('./images/'+sample[0], './ChestX8_lf/images/' + sample[0])
    