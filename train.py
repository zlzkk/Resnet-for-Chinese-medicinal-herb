# coding=gb2312
import os


import torchvision
from torch.utils.data import DataLoader
import dataset
import copy
import time
import torch
import matplotlib.pyplot as plt
from model_utils import resnet18, ConvNeXt
from model_utils import GoogLeNet

import torch.nn as nn
import pandas as pd

from tqdm import tqdm
import numpy as np
from sklearn import metrics
from test import Singletask_test_model_save_results
epochs = 2000
batch_size = 16
train_dir = '../data'
train_dir_csv = '../data/data1/train.csv'
test_dir_csv = '../data/data1/test.csv'
model_path = '../out'
task_names = 'Task_Chinese_medicinal_herb'
sep=','
use_gpu = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(use_gpu)

def Singletask_train_model_save_results(model,criterion,dataloader,optimizer,dataloader1,weight_scheduler,device=None):
    print("Training the model and saving the results:", flush=True)
    min1 = 0
    model.train()
    for i_epoch in range(epochs):
            loss_history = []
            probs_ = []
            preds_ = []
            labels_ = []

            running_corrects = 0
            count = 0

            for idx, (inputs, labels) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Train Iteration"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                # outputs = torch.nn.functional.softmax(outputs, 1)
                loss = criterion(outputs , labels)
                outputs = torch.nn.functional.softmax(outputs, 1)
                probs, preds = torch.max(outputs, 1)

                preds_.extend(preds.detach().cpu().numpy().tolist())
                probs_.extend(probs.detach().cpu().numpy().tolist())
                labels_.extend(labels.detach().data.cpu().numpy().tolist())
                # print(preds)
                # print(labels.data)
                running_corrects += torch.sum(preds == labels.data)
                count += len(labels.data)
                loss.backward()
                optimizer.step()
                loss_history.append([loss.item(), 0.])
            f1_score = metrics.f1_score(labels_, preds_, average="macro")
            print(f1_score)
            print(f'the epoch is {i_epoch}')

            epoch_losses = np.mean(np.array(loss_history), axis=0)


            sore = Singletask_test_model_save_results(model, dataloader1, i_epoch,device=device)

            if min1 < sore:
                min1 = sore
                print("super", epoch_losses[0])
                torch.save(model.state_dict(), 'goodrescov0.87.pth')
            weight_scheduler.step()
    # # #
    # torch.save(model.state_dict(), 'endonlybasic.pth')

def main():
    data_transforms = {
        'train': torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 224)),
            torchvision.transforms.CenterCrop((256, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    data_transforms = {
        'train': torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 224)),
            torchvision.transforms.RandomHorizontalFlip(),  # 随机水平翻转
            torchvision.transforms.RandomRotation(degrees=15),  # 随机旋转
            torchvision.transforms.CenterCrop((256, 224)), # 随机裁剪并缩放
            torchvision.transforms.ToTensor(),
            torchvision. transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test':  torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 224)),

            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    # image_datasets = { 'train': dataset.SingleTaskDataset(train_dir_csv,sep, train_dir, data_transforms['train'], task_names)}
    train_dataset = dataset.SingleTaskDataset(train_dir_csv,task_names,sep, train_dir, data_transforms['train'])
    test_dataset = dataset.SingleTaskDataset(test_dir_csv,task_names,sep, train_dir, data_transforms['test'])
    # # model = resnet18(num_classes=14, include_top=True)
    # model =ConvNeXt(num_classes=14)
    model = GoogLeNet(num_classes=14)
    optimizer = torch.optim.Adam(model.parameters(), lr=10 ** -5.5, betas=(0.5, 0.999), eps=1e-6, weight_decay=1e-5)
    weight_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.5)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=80)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=80)

    Singletask_train_model_save_results(model,criterion,train_loader,optimizer,test_loader,weight_scheduler,device=device)
if __name__ == "__main__":
    main()