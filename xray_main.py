import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import sys

import argparse

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


from DatasetGenerator import *
from xray_finetune import fine_tune_ViT, test

import clip

# copied from https://github.com/openai/CLIP/issues/83

os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"

class Model(nn.Module):
    def __init__(self, embed_dim, n_classes, name):
        super().__init__()
        clip_model, preprocess = clip.load(name, device=device, jit=False)   # jit=False
        # clip_model, preprocess = clip.load(name, device=device, jit=True)
        classifier = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(embed_dim, n_classes)
        )
        # classifier = nn.Linear(embed_dim, n_classes)
        self.base_model = clip_model
        self.classifier = classifier
        # self.preprocess = preprocess
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # return self.sigmoid(self.classifier(self.base_model.encode_image(x)))
        return self.classifier(self.base_model.encode_image(x))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='finetune CLIP')
    # 'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14'
    parser.add_argument('--model', default='ViT-B/32', help='model name') 
    parser.add_argument('--lr', default=1e-6, type=float, help='learning rate')
    parser.add_argument('--e', default=100, type=int, help='training epochs')
    parser.add_argument('--wd', type=float, default=0.001, help='weight decay')
    parser.add_argument('--eps', type=float, default=1e-6, help='eps')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--name', default='st_jitT', help='experiment_name')
    parser.add_argument('--rootPath', default='../chestxray', help='dataset root path')
    parser.add_argument('--train', action='store_true', help='to train or not')
    # parser.add_argument('--train', default=True, help='to train or not')
    parser.add_argument('--test', action='store_true', help='to test or not')
    # parser.add_argument('--test', default=True, help='to test or not')

    args = parser.parse_args()

    print('parameter settings:\n')
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))

    BATCH_SIZE = args.batch_size
    n_classes = 14
    if args.model == 'ViT-B/32':
        embed_dim = 512
    elif args.model == 'ViT-L/14':
        embed_dim = 768
    elif args.model == 'RN50':
        embed_dim = 1024

    pathDirData = args.rootPath
    pathFileTrain = './mine/myChestxray/dataset/train_1.txt'
    pathFileVal = './mine/myChestxray/dataset/val_1.txt'
    pathFileTest = './mine/myChestxray/dataset/test_1.txt'

    # embed_dim = 1
    device = "cuda" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    model = Model(embed_dim, n_classes, args.model)
    _, preprocess = clip.load(args.model, device=device, jit=False)


    train_set = DatasetGenerator(pathDirData, pathFileTrain, preprocess) 
    val_set = DatasetGenerator(pathDirData, pathFileVal, preprocess)
    test_set = DatasetGenerator(pathDirData, pathFileTest, preprocess)
    print("Train data length:", len(train_set))
    print("Valid data length:", len(val_set))
    print("Test data length:", len(test_set))

    train_dataloader= DataLoader(dataset = train_set, batch_size = BATCH_SIZE,
                                shuffle=True, num_workers=0, pin_memory=True)
    val_dataloader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, 
                            shuffle=False, num_workers=0, pin_memory=True)
    test_dataloader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, 
                            num_workers=0, pin_memory=True)

    

    if device == "cpu":
        model.float()
    else:
        clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

    model = nn.DataParallel(model)

    if args.train == True:
        model = fine_tune_ViT(model, train_dataloader, val_dataloader, args, dev=device, multi_label_data=True)
    elif args.test == True:
        test(model, test_dataloader, args, dev=device, multi_label_data=True)



