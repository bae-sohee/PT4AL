'''Train MVTec with PyTorch.'''
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

from models import *
from loader import Loader, RotationLoader, MvtecLoader
from utils import progress_bar
import numpy as np

import torch, gc
gc.collect()
torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description='PyTorch mvtec Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize(256),  
    # transforms.Grayscale(num_output_channels=3), #grid, screw, zipper
    transforms.ToTensor(),
    transforms.Normalize([0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629]),
])

transform_test = transforms.Compose([
    transforms.Resize(256), 
    # transforms.Grayscale(num_output_channels=3), #grid, screw, zipper
    transforms.ToTensor(),
    transforms.Normalize([0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629]),
])

trainset = MvtecLoader(is_train=True, transform=transform_test)
n_train_batches = 10
train_batch_size = len(trainset) // n_train_batches
trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)

testset = MvtecLoader(is_train=False,  transform=transform_test)
n_test_batches = 10
test_batch_size = len(testset) // n_test_batches
testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)


# Model
print('==> Building model..')
# net = ResNet18()
# net.linear = nn.Linear(512, 3)
# net = net.to(device)
import torchvision.models as models
# Load the pretrained ResNet-18 model
pretrained_resnet = models.resnet18(pretrained=True)

# Modify the final fully connected layer to match the desired number of classes
pretrained_resnet.fc = nn.Linear(pretrained_resnet.fc.in_features, 3)  # Assuming 3 classes

# Move the model to the desired device (CPU or GPU)
pretrained_resnet = pretrained_resnet.to(device)

if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    net = torch.nn.DataParallel(pretrained_resnet)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90])

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, inputs1, inputs2, targets, targets1, targets2) in enumerate(trainloader):
        inputs, inputs1, targets, targets1 = inputs.to(device), inputs1.to(device), targets.to(device), targets1.to(device)
        inputs2, targets2 = inputs2.to(device), targets2.to(device)
        optimizer.zero_grad()
        outputs, outputs1, outputs2 = net(inputs), net(inputs1), net(inputs2)

        loss1 = criterion(outputs, targets)
        loss2 = criterion(outputs1, targets1)
        loss3 = criterion(outputs2, targets2)
        loss = (loss1+loss2+loss3)/3.
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        _, predicted1 = outputs1.max(1)
        _, predicted2 = outputs2.max(1)
        total += targets.size(0)*3

        correct += predicted.eq(targets).sum().item()
        correct += predicted1.eq(targets1).sum().item()
        correct += predicted2.eq(targets2).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


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

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    with open('./best_cutpaste.txt','a') as f:
        f.write(str(acc)+':'+str(epoch)+'\n')
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        # save rotation weights
        torch.save(state, './checkpoint/cutpaste.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+15):
    train(epoch)
    test(epoch)
    scheduler.step()
