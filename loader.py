import glob
import os
from PIL import Image, ImageFilter

from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
import numpy as np
import random
import cv2
import torchvision.transforms.functional as F

from CutPaste import CutPaste

# CutPaste Object

make_cutpaste = CutPaste()

num_classes = 2                                               #########################

class RotationLoader(Dataset):
    def __init__(self, is_train=True, transform=None, path='./DATA'):
        self.is_train = is_train
        self.transform = transform
        # self.h_flip = transforms.RandomHorizontalFlip(p=1)
        if self.is_train == 0: # train
            self.img_path = glob.glob('./DATA/train/*/*')
        else:
            self.img_path = glob.glob('./DATA/train/*/*')

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        img = Image.fromarray(img)

        if self.is_train:
            img = self.transform(img)
            img1 = torch.rot90(img, 1, [1,2])
            img2 = torch.rot90(img, 2, [1,2])
            img3 = torch.rot90(img, 3, [1,2])
            imgs = [img, img1, img2, img3]
            rotations = [0,1,2,3]
            random.shuffle(rotations)
            return imgs[rotations[0]], imgs[rotations[1]], imgs[rotations[2]], imgs[rotations[3]], rotations[0], rotations[1], rotations[2], rotations[3]
        else:
            img = self.transform(img)
            img1 = torch.rot90(img, 1, [1,2])
            img2 = torch.rot90(img, 2, [1,2])
            img3 = torch.rot90(img, 3, [1,2])
            imgs = [img, img1, img2, img3]
            rotations = [0,1,2,3]
            random.shuffle(rotations)
            return imgs[rotations[0]], imgs[rotations[1]], imgs[rotations[2]], imgs[rotations[3]], rotations[0], rotations[1], rotations[2], rotations[3], self.img_path[idx]

class Loader2(Dataset):
    def __init__(self, is_train=True, transform=None, path='./DATA', path_list=None):
        self.is_train = is_train
        self.transform = transform
        self.path_list = path_list
        self.label_dict = {'good': 0, 'defective' :1}  #Initialize a dictionary that maps class labels to numbers 
        ############################## 

        if self.is_train: # train
            self.img_path = path_list
        else:
            if path_list is None:
                self.img_path = glob.glob('./DATA/train/*/*') # for loss extraction
            else:
                self.img_path = path_list
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        if self.is_train:
            img = cv2.imread(self.img_path[idx][:-1])
        else:
            if self.path_list is None:
                img = cv2.imread(self.img_path[idx])
            else:
                img = cv2.imread(self.img_path[idx][:-1])
        img = Image.fromarray(img)
        img = self.transform(img)
        label = self.label_dict[self.img_path[idx].split('/')[-2]]

        return img, label
    
class Loader_Cold(Dataset):
    def __init__(self, is_train=True, transform=None, path='./DATA'):
        self.classes = num_classes
        self.is_train = is_train
        self.transform = transform
        with open('./loss/batch_9.txt', 'r') as f: 
            self.list = f.readlines()
        self.list = [self.list[i*5] for i in range(1000)]
        if self.is_train==True: # train
            self.img_path = self.list
        else:
            self.img_path = glob.glob('./DATA/test/*/*')

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        if self.is_train ==True:
            img = cv2.imread(self.img_path[idx][:-1])
        else:
            img = cv2.imread(self.img_path[idx])
        img = Image.fromarray(img)
        img = self.transform(img)
        label = int(self.img_path[idx].split('/')[-2])

        return img, label
    
class Loader(Dataset):
    def __init__(self, is_train=True, transform=None, path='./DATA'):
        self.classes = num_classes
        self.is_train = is_train
        self.transform = transform
        self.label_dict = { 'good': 0, 'defective' :1}  #Initialize a dictionary that maps class labels to numbers 
                                    ##############################

        if self.is_train: # train
            self.img_path = glob.glob('./DATA/train/*/*')
        else:
            self.img_path = glob.glob('./DATA/test/*/*')

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        img = Image.fromarray(img)
        img = self.transform(img)
        label = self.label_dict[self.img_path[idx].split('/')[-2]]

        return img, label

class MvtecLoader(Dataset):
    def __init__(self, is_train=True, transform=None, path='./DATA'):
        self.is_train = is_train
        self.transform = transform
        self.make_cutpaste = CutPaste

        if self.is_train == 0: # train
            self.img_path = glob.glob('./DATA/train/*/*')
        else:
            self.img_path = glob.glob('./DATA/train/*/*')
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        img = Image.fromarray(img)

        img1 = np.array(make_cutpaste.cutpaste(Image.open(self.img_path[idx])))
        img2 = np.array(make_cutpaste.cutpaste(Image.open(self.img_path[idx])))

        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)

        # cutpaste coded needed here
        if self.is_train:
            img = self.transform(img)
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            imgs = [img, img1, img2]
            cutpates = [0,1,2]
            random.shuffle(cutpates)
            return imgs[cutpates[0]], imgs[cutpates[1]], imgs[cutpates[2]], cutpates[0], cutpates[1], cutpates[2]
        
        else:
            img = self.transform(img)
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            imgs = [img, img1, img2]
            cutpates = [0,1,2]
            random.shuffle(cutpates)
            return imgs[cutpates[0]], imgs[cutpates[1]], imgs[cutpates[2]], cutpates[0], cutpates[1], cutpates[2],  self.img_path[idx]
