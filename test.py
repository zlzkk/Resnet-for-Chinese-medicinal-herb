# coding=gb2312
import os
import argparse
import random

import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn import metrics
from tqdm import tqdm
import dataset
from model_utils import resnet18, ConvNeXt
from model_utils import GoogLeNet
'''
    test.py���޸ĵĲ��ֽ��Ĳ��֣�
        1��from model_utils import resnet18  # import��ģ�Ϳɸ���ʵ���޸�
        2��parser.add_argument�����в�����ֻ���޸� default �ֶ�
        3��main�������ֵ�ģ�͵��ã�model = resnet18(num_classes=total_classes, include_top=True)  # �ɸ���ʵ�ʵ���ģ��
        4�����ݼ��ز��֣�image_datasets��test_dataset
'''


parser = argparse.ArgumentParser(description='Classification Competition Project')
parser.add_argument('--name', default='multitask_resnet', type=str, help='name of the run')
parser.add_argument('--seed', default=2, type=int, help='random seed')
parser.add_argument('--test',
                    default='../data/data1/test.csv',
                    type=str, help='path to test image files/labels')
parser.add_argument('--sep', default=',', type=str, help='column separator used in csv(default: ",")')
parser.add_argument('--data-dir', default='../data',
                    type=str, help='root directory of images')  # images���Ŀ¼(�����Ŀ¼)��ע��ο�csv�е�image_path�������ȡ���ɹ�
parser.add_argument('--best-state-path',
                    default='./goodrescov0.87.pth',
                    type=str,
                    help='path to load best state')
parser.add_argument('--batch-size', default=8, type=int, help='batch size (default: 32)')
parser.add_argument('--task-name', default='Task_Chinese_medicinal_herb', type=str,
                    help='names of the task separated y comma')
parser.add_argument('--task-names', default='tongue_colour,moss_colour,tongue_shape,moss_quality', type=str,
                    help='names of the task separated y comma')

num_classes_tasks = {
    'tongue_colour': 5,
    'moss_colour': 3,
    'tongue_shape': 6,
    'moss_quality': 9,
    'Task_Chinese_medicinal_herb': 14
}

def calculate_accuracy(y_true, y_pred):
    correct_predictions = sum(y_true == y_pred)
    accuracy = correct_predictions / len(y_true)
    return accuracy
def calculate_precision(y_true, y_pred):
    TP = sum((y_true == 1) & (y_pred == 1))
    FP = sum((y_true == 0) & (y_pred == 1))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    return precision
def calculate_recall(y_true, y_pred):
    TP = sum((y_true == 1) & (y_pred == 1))
    FN = sum((y_true == 1) & (y_pred == 0))
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    return recall
from sklearn.metrics import roc_auc_score

def calculate_auc_roc(y_true, y_prob):
    auc = roc_auc_score(y_true, y_prob)
    return auc
from sklearn.metrics import confusion_matrix

def calculate_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm

def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def Multitask_test_model_save_results(model, dataloader, num_classes, class_names=None, device=None):
    print("Testing the model and saving the results:", flush=True)
    model.eval()

    num_tasks = len(num_classes)
    preds_ = [[] for _ in range(num_tasks)]
    labels_ = [[] for _ in range(num_tasks)]
    with torch.no_grad():
        for _, (inputs, labels) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Test Iteration"):
            inputs = inputs.to(device)
            labels = [labels[i].to(device) for i in range(len(labels))]
            outputs = model(inputs)
            idx_start = 0
            for idx in range(num_tasks):
                output_idx = outputs[:, idx_start:idx_start + num_classes[idx]]
                label_idx = labels[idx]
                valid_idx = torch.nonzero(1 * (label_idx != -1)).view(-1)
                if len(valid_idx) == 0:
                    continue
                idx_start += num_classes[idx]
                _, preds = torch.max(output_idx[valid_idx], 1)
                preds_[idx].extend(preds.cpu().numpy().tolist())
                labels_[idx].extend(label_idx[valid_idx].data.cpu().numpy().tolist())

    f1_scores = []
    for idx in range(num_tasks):
        label_y = []
        label_pred = []
        task_class_names = class_names[idx]
        for i in range(len(preds_[idx])):
            label_y.append(task_class_names[labels_[idx][i]])
            label_pred.append(task_class_names[preds_[idx][i]])
        f1_score = metrics.f1_score(label_y, label_pred, average="macro")
        f1_scores.append(f1_score)
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    file_name = f"{parent_dir}/out/multi_output.txt"
    sum_f1_score = sum(f1_scores)
    average_f1_score = sum_f1_score/num_tasks
    print("f1ֵΪ:", average_f1_score)
    with open(file_name, "a") as file:
        file.write(str(average_f1_score)+"\n")
    print("���Խ���ѱ��浽", file_name)


def Singletask_test_model_save_results(model, dataloader, epoch,device=None):
    print("Testing the model and saving the results:", flush=True)

    model.eval()
    probs_ = []
    preds_ = []
    labels_ = []

    running_corrects = 0
    count = 0
    with torch.no_grad():
        for idx, (inputs, labels) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Test Iteration"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            outputs = torch.nn.functional.softmax(outputs, 1)
            probs, preds = torch.max(outputs, 1)

            preds_.extend(preds.cpu().numpy().tolist())
            probs_.extend(probs.cpu().numpy().tolist())
            labels_.extend(labels.data.cpu().numpy().tolist())
            print(preds)
            print(labels.data)
            running_corrects += torch.sum(preds == labels.data)
            count += len(labels.data)
    f1_score = metrics.f1_score(labels_, preds_, average="macro")
    print(f1_score)
    accuracy = metrics.accuracy_score(labels_, preds_)
    precision = metrics.precision_score(labels_, preds_, average="macro")
    recall = metrics.recall_score(labels_, preds_, average="macro")

    # ��ӡ����ָ��
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    file_name = f"{parent_dir}/out/single_outputgoole.txt"
    epoch = epoch
    with open(file_name, "a") as file:
        file.write(str(f1_score) +"\n")
    print("���Խ���ѱ��浽", file_name)
    return f1_score


def main():
    global device
    use_gpu = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(use_gpu)
    print("Using device: {}".format(use_gpu), flush=True)
    args = parser.parse_args()
    print(args, flush=True)
    set_seed(args.seed)
    sep = args.sep
    data_dir = args.data_dir
    best_state_path = args.best_state_path
    batch_size = args.batch_size
    task_name = args.task_name
    task_names = args.task_names.split(',')
    num_classes = [num_classes_tasks[x] for x in task_names]
    total_classes = 14
    # model = resnet18(num_classes=total_classes, include_top=True)  # �ɸ���ʵ�ʵ���ģ��
    #
    # model = ConvNeXt(num_classes=14)
    model = GoogLeNet(num_classes=14)
    best_state = torch.load(best_state_path)
    model.load_state_dict(best_state)

    data_transforms = {
        'test': torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 224)),
            torchvision.transforms.CenterCrop((256, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    print("Initializing Datasets and Dataloaders...", flush=True)
    image_datasets = {
                    'test': dataset.SingleTaskDataset(args.test, task_name, sep, data_dir, data_transforms['test'])}
                    # 'test': dataset.MultitaskDataset(args.test, sep, data_dir, data_transforms['test'], task_names)}

    class_names = image_datasets['test'].classes
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use {} GPUs!".format(torch.cuda.device_count()), flush=True)
        model = torch.nn.DataParallel(model)

    if args.test:
        test_dataset = dataset.SingleTaskDataset(args.test, task_name, sep, data_dir, data_transforms['test'])
        # test_dataset = dataset.MultitaskDataset(args.test, sep, data_dir, data_transforms['test'], task_names)
        print("There are {} test images.".format(len(test_dataset)))
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=80)
        # Multitask_test_model_save_results(model, test_dataloader, num_classes, class_names = class_names,
        #                                   device = device)
        Singletask_test_model_save_results(model, test_dataloader,1, device=device)


if __name__ == "__main__":
    main()