'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import random
import numpy as np

from models import *
from loader import Loader, RotationLoader, MvtecLoader
from utils import progress_bar

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


transform_test = transforms.Compose([transforms.Resize((256, 256)),
                                    # transforms.Grayscale(num_output_channels=3), #grid, screw, zipper
                                    transforms.ToTensor(), 
                                    transforms.Normalize([0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629]),
])

testset = MvtecLoader(is_train=False,  transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=8)

# net = ResNet18()
# net.linear = nn.Linear(512, 4)
# net = net.to(device)

import torchvision.models as models
pretrained_resnet = models.resnet18(pretrained=True)
pretrained_resnet.fc = nn.Linear(pretrained_resnet.fc.in_features, 3)                # cutpaste 3
pretrained_resnet = pretrained_resnet.to(device)


if device == 'cuda':
    net = torch.nn.DataParallel(pretrained_resnet)
    cudnn.benchmark = True

checkpoint = torch.load('./checkpoint/cutpaste.pth')                                    ##################################

net.load_state_dict(checkpoint['net'])

criterion = nn.CrossEntropyLoss()

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
            
    with torch.no_grad():
        for batch_idx, (inputs, inputs1, inputs2, targets, targets1, targets2, path) in enumerate(testloader):
            inputs, inputs1, targets, targets1 = inputs.to(device), inputs1.to(device), targets.to(device), targets1.to(device)
            inputs2, targets2 = inputs2.to(device), targets2.to(device)
            outputs = net(inputs)
            outputs1 = net(inputs1)
            outputs2 = net(inputs2)
            loss1 = criterion(outputs, targets)
            loss2 = criterion(outputs1, targets1)
            loss3 = criterion(outputs2, targets2)
            loss = (loss1+loss2+loss3)/3.
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            _, predicted1 = outputs1.max(1)
            _, predicted2 = outputs2.max(1)
            total += targets.size(0)*3

            correct += predicted.eq(targets).sum().item()
            correct += predicted1.eq(targets1).sum().item()
            correct += predicted2.eq(targets2).sum().item()
                        
            loss = loss.item()
            s = str(float(loss)) + '_' + str(path[0]) + "\n"

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))      
              
            with open('./cutpaste_loss.txt', 'a') as f:                         
                f.write(s)
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

if __name__ == "__main__":
    test(1)

    with open('./cutpaste_loss.txt', 'r') as f:                                                   ###############################
        losses = f.readlines()
    
    loss_1 = []
    name_2 = []

    for j in losses:
        loss_1.append(j[:-1].split('_')[0])
        name_2.append(j[:-1].split('_',1))

    s = np.array(loss_1)
    sort_index = np.argsort(s)
    x = sort_index.tolist()
    x.reverse()
    sort_index = np.array(x)  # convert to high loss first

    if not os.path.isdir('loss'):
        os.mkdir('loss')

    # Calculate the batch size based on the total number of data
    total_data = len(losses)
    batch_size = total_data // 5                                        ########### batch size 

    for i in range(5):                                                  ###########
        # sample minibatch from unlabeled pool
        start = i * batch_size
        # handle the case for the last batch which might have less than batch_size items
        end = (i + 1) * batch_size if i < 4 else total_data

        sample = sort_index[start:end]
        b = {}

        for jj in sample:
            second_item = name_2[jj][1]
            split_string  = second_item.split('/')  
            class_name = split_string[3] 
            if class_name not in b:
                b[class_name] = 0
            b[class_name] += 1  # Increase the count for the class

        print(f'{i} batch Class Distribution: {b}')

        s = './loss/batch_' + str(i) + '.txt'
        for k in sample:
            with open(s, 'a') as f:
                f.write(str(name_2[k][1]) + '\n')
