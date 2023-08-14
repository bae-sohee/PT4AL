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
import numpy as np

from models import *
from loader import Loader, Loader2
from utils import progress_bar

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np

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
                                      transforms.Grayscale(num_output_channels=3), #grid, screw, zipper
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629])])

transform_test = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.Grayscale(num_output_channels=3), #grid, screw, zipper 
                                    transforms.ToTensor(), 
                                    transforms.Normalize([0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629])])


testset = Loader(is_train=False,  transform=transform_test)
n_test_batches = 10
test_batch_size = len(testset) // n_test_batches
testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)


classes =('good','defective')                   #########################
           # ("good","anomaly")                                      # binary
           # ("good","broken_large","broken_small","contamination")  # bottle
           # ('cut_inner_insulation', 'good', 'missing_cable', 'cut_outer_insulation', 'poke_insulation', 'missing_wire', 'bent_wire', 'cable_swap', 'combined')   # cable
           # ('faulty_imprint', 'squeeze', 'poke', 'good', 'scratch', 'crack') # "capsule"
           # ('good', 'metal_contamination', 'color', 'cut', 'thread', 'hole') # "carpet"
           # ('broken', 'good', 'metal_contamination', 'glue', 'thread', 'bent') #grid
           # ('print', 'good', 'cut', 'crack', 'hole') # "hazelnut"
           # ('good', 'color', 'flip', 'scratch', 'bent') # "metal_nut"
           # ('scratch_neck', 'good', 'thread_top', 'scratch_head', 'manipulated_front', 'thread_side')# screw
           # ('rough', 'good', 'fabric_border', 'fabric_interior', 'squeezed_teeth', 'split_teeth', 'broken_teeth', 'combined') # "zipper"
           # ('poke', 'good', 'color', 'glue', 'cut', 'fold') # leather
           # ('faulty_imprint', 'good', 'color', 'scratch', 'contamination', 'crack', 'pill_type', 'combined') # pil
           # ('rough', 'glue_strip', 'good', 'crack', 'gray_stroke', 'oil')                     # tile
           # ('good','defective')                   # toothbrush
           #  ('cut_lead', 'good', 'bent_lead', 'misplaced', 'damaged_case')   #transistor
           # (good', 'color', 'scratch', 'hole', 'combined', 'liquid') wood
           
# classes = ("bottle","cable","capsule","hazelnut","metal_nut",
#            "pill","screw","toothbrush","transistor","zipper",
#            "carpet","grid","leather","tile","wood",)

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

# Training
def train(net, criterion, optimizer, epoch, trainloader):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        # print("Inputs shape:", inputs.shape)
        outputs = net(inputs)
        # print("Outputs shape:", outputs.shape)
        # print("Outputs:", outputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(net, criterion, epoch, cycle):
    global best_acc
    global best_auroc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    # #(multi)
    # all_targets = []
    # all_outputs = []
    #(binary)
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

            # binary auroc

            all_predicted_scores.extend(outputs[:, 1].cpu().numpy())  # Assuming the positive class is index 1
            all_targets.extend(targets.cpu().numpy())

            # #multi auroc

            # all_outputs.extend(outputs.cpu().numpy())
            # all_targets.extend(targets.cpu().numpy())

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Calculate the ROC AUC score for all predictions (binary)
    roc_score = roc_auc_score(all_targets, all_predicted_scores)

    # # Calculate the ROC AUC score for all predictions (multi)
    # roc_score = roc_auc_score(all_targets, all_probs, multi_class='ovr')     
    
    # # 레이블을 one-hot encoding 형식으로 변환합니다.
    # n_classes = len(set(all_targets))
    # all_targets_onehot = label_binarize(all_targets, classes=list(range(n_classes)))

    # # ROC AUC score
    # print("AUROC score : %.3f" % roc_score)
            
    # # AUROC를 계산합니다.
    # roc_score = roc_auc_score(all_targets_onehot, all_outputs, average='macro')


    print(f"Epoch: {epoch}, AUROC: {roc_score:.4f}")
    
    # Save checkpoint acc.
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
        torch.save(state, f'./checkpoint/main_acc_{cycle}.pth')
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
        torch.save(state, f'./checkpoint/main_auroc_{cycle}.pth')  
        best_auroc = roc_score  

# class-balanced sampling (pseudo labeling)
def get_plabels(net, samples, cycle):
    # dictionary with 10 keys as class labels
    class_dict = {}
    [class_dict.setdefault(x,[]) for x in range(num_classes)]                         ############################

    sub5k = Loader2(is_train=False,  transform=transform_test, path_list=samples)
    ploader = torch.utils.data.DataLoader(sub5k, batch_size=1, shuffle=False, num_workers=2)

    # overflow goes into remaining
    remaining = []
    net.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(ploader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            if len(class_dict[predicted.item()]) < 100:
                class_dict[predicted.item()].append(samples[idx])
            else:
                remaining.append(samples[idx])
            progress_bar(idx, len(ploader))

    sample1k = []
    for items in class_dict.values():
        if len(items) == 100:
            sample1k.extend(items)
        else:
            # supplement samples from remaining 
            sample1k.extend(items)
            add = 100 - len(items)
            sample1k.extend(remaining[:add])
            remaining = remaining[add:]
    
    return sample1k

# confidence sampling (pseudo labeling)
## return 1k samples w/ lowest top1 score
def get_plabels2(net, samples, cycle):
    # dictionary with 10 keys as class labels
    class_dict = {}
    [class_dict.setdefault(x,[]) for x in range(num_classes)]                                ########################

    sample1k = []
    sub5k = Loader2(is_train=False,  transform=transform_test, path_list=samples)
    ploader = torch.utils.data.DataLoader(sub5k, batch_size=1, shuffle=False, num_workers=2)

    top1_scores = []
    net.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(ploader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            scores, predicted = outputs.max(1)
            # save top1 confidence score 
            outputs = F.normalize(outputs, dim=1)
            probs = F.softmax(outputs, dim=1)
            top1_scores.append(probs[0][predicted.item()].cpu().numpy())  # CUDA 텐서를 Numpy로 변환
            progress_bar(idx, len(ploader))
    idx = np.argsort(top1_scores)
    samples = np.array(samples)

    # Calculate the number of samples that corresponds to 50% of the sample size
    k_samples = int(len(samples) * 0.5)
    return samples[idx[:k_samples]]

# entropy sampling
def get_plabels3(net, samples, cycle):
    sample1k = []
    sub5k = Loader2(is_train=False,  transform=transform_test, path_list=samples)
    ploader = torch.utils.data.DataLoader(sub5k, batch_size=1, shuffle=False, num_workers=2)

    top1_scores = []
    net.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(ploader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            e = -1.0 * torch.sum(F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1), dim=1)
            top1_scores.append(e.view(e.size(0)))
            progress_bar(idx, len(ploader))
    idx = np.argsort(top1_scores)
    samples = np.array(samples)
    return samples[idx[-1000:]]

def get_classdist(samples):
    class_dist = np.zeros(num_classes)                                                      #########################
    for sample in samples:
        label = int(sample.split('/')[-2])
        class_dist[label] += 1
    return class_dist

if __name__ == '__main__':
    labeled = []
        
    CYCLES = 5                                                                              # batch size
    for cycle in range(CYCLES):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[160])

        best_acc = 0
        best_auroc = 0 
        print('Cycle ', cycle)

        # open 5k batch (sorted low->high)
        with open(f'./loss/batch_{cycle}.txt', 'r') as f:
            samples = f.readlines()
            
        if cycle > 0:
            print('>> Getting previous checkpoint')
            # prevnet = ResNet18().to(device)
            # prevnet = torch.nn.DataParallel(prevnet)
            checkpoint = torch.load(f'./checkpoint/main_acc_{cycle-1}.pth')
            checkpoint_auroc = torch.load(f'./checkpoint/main_auroc_{cycle-1}.pth')
            net.load_state_dict(checkpoint['net'])
            net.load_state_dict(checkpoint_auroc['net'])                            

            sample1k = get_plabels2(net, samples, cycle)
        else:
            # first iteration: sample 1k at even intervals
            samples = np.array(samples)

            sample_size = len(samples)  # sample 수
            print(sample_size)
            k_sample_size = sample_size *0.5 # sample * 0.5

            interval = sample_size // k_sample_size
            indices = np.arange(0, sample_size, interval)
            sample1k = [samples[int(i)] for i in indices]  # uniform sampling
            print(len(sample1k))

        # add 1k samples to labeled set
        labeled.extend(sample1k)
        print(f'>> Labeled length: {len(labeled)}')
        trainset = Loader2(is_train=True, transform=transform_train, path_list=labeled)
        n_train_batches = 10
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=n_train_batches, shuffle=True, num_workers=2)

        for epoch in range(200):
            train(net, criterion, optimizer, epoch, trainloader)
            test(net, criterion, epoch, cycle)
            scheduler.step()

        with open(f'./main_best_acc.txt', 'a') as f:
            f.write(str(cycle) + ' ' + str(best_acc)+'\n')
        with open(f'./main_best_auroc.txt', 'a') as f:
            f.write(str(cycle) + ' ' + str(best_auroc)+'\n')
         