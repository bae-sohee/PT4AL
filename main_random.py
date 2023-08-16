# cold start ex
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import random

from models import *
from utils import progress_bar
from loader import Loader, Loader2

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

parser = argparse.ArgumentParser(description='PyTorch MVTec Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_auroc = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# MVTec Data
print('==> Preparing data..')
transform_train = transforms.Compose([transforms.Resize((256, 256)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629])])

transform_test = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.ToTensor(), 
                                    transforms.Normalize([0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629])])

trainset = Loader(is_train=True, transform=transform_train)

#random sampling
indices = list(range(len(trainset)))
random.shuffle(indices)
labeled_set = indices[:len(trainset) // 10]  

n_train_batches = 10
train_batch_size = len(trainset) // n_train_batches
trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, num_workers=2, sampler=SubsetRandomSampler(labeled_set))

testset = Loader(is_train=False, transform=transform_test)
n_test_batches = 10
test_batch_size = len(testset) // n_test_batches
testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)


classes = ('good','defective')
# classes = ('good', 'color', 'scratch', 'hole', 'combined', 'liquid') 
num_classes = len(classes)
# Model
print('==> Building model..')
# net = ResNet18()
# net = net.to(device)

import torchvision.models as models
pretrained_resnet = models.resnet18(pretrained=True)
pretrained_resnet.fc = nn.Linear(pretrained_resnet.fc.in_features, num_classes)  # Assuming 4 classes
pretrained_resnet = pretrained_resnet.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(pretrained_resnet)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[160])

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    global best_auroc

    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    #binary auroc

    all_predicted_scores = []  # Store predicted scores for AUROC calculation
    all_targets = []  # Store true labels for AUROC calculation   
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            #binary auroc
            all_predicted_scores.extend(outputs[:, 1].cpu().numpy())  # Assuming the positive class is index 1
            all_targets.extend(targets.cpu().numpy())

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # Calculate the ROC AUC score for all predictions (binary)
    roc_score = roc_auc_score(all_targets, all_predicted_scores)

    # #(multi)
    # all_targets = []
    # all_outputs = []

    # with torch.no_grad():
    #     for batch_idx, (inputs, targets) in enumerate(testloader):
    #         inputs, targets = inputs.to(device), targets.to(device)
    #         outputs = net(inputs)
    #         loss = criterion(outputs, targets)

    #         test_loss += loss.item()
    #         _, predicted = outputs.max(1)
    #         total += targets.size(0)
    #         correct += predicted.eq(targets).sum().item()

    #         #multi auroc

    #         all_outputs.extend(outputs.cpu().numpy())
    #         all_targets.extend(targets.cpu().numpy())

    #         progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
    #                      % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # # 레이블을 one-hot encoding 형식으로 변환합니다.
    # n_classes = len(set(all_targets))
    # all_targets_onehot = label_binarize(all_targets, classes=list(range(n_classes)))
    # # AUROC를 계산합니다.
    # roc_score = roc_auc_score(all_targets_onehot, all_outputs, average='macro')

    print(f"Epoch: {epoch}, AUROC: {roc_score:.4f}")

    # # ROC AUC score
    # print("AUROC score : %.3f" % roc_score)
            
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/acc_random.pth')
        best_acc = acc

    # Save checkpoint auroc.
    if roc_score > best_auroc:  
        print('Saving AUROC checkpoint..')
        state = {
            'net': net.state_dict(),
            'auroc': roc_score,  
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'./checkpoint/auroc_random.pth')  
        best_auroc = roc_score  

        print('>> Getting previous checkpoint')

        checkpoint = torch.load(f'./checkpoint/acc_random.pth')
        checkpoint_auroc = torch.load(f'./checkpoint/auroc_random.pth')
        net.load_state_dict(checkpoint['net'])
        net.load_state_dict(checkpoint_auroc['net'])    

for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()


with open(f'./main_best_acc.txt', 'a') as f:
    f.write('random first acc'+ ' ' + str(best_acc)+'\n')
with open(f'./main_best_auroc.txt', 'a') as f:
    f.write('random first auroc' + ' ' + str(best_auroc)+'\n')