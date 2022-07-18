import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import shutil
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset

# we added the separation and header because the data is not organized, one can run without sep and header
#to understand the difference

train_df = pd.read_csv('/data2/yinuo/covidx/train.txt', sep=" ", header=None)

#Columns are added because it was seen that column names were 0,1,2,3, so new column names are added
#which are given in descriptions
train_df.columns=['patient id', 'filename', 'class', 'data source']

# Since we are doing image classification, patient id and data source is of no importance to us, so
#we cn drop them
train_df=train_df.drop(['patient id', 'data source'], axis=1 )

#same as train
test_df = pd.read_csv('/data2/yinuo/covidx/test.txt', sep=" ", header=None)
test_df.columns=['id', 'filename', 'class', 'data source' ]
test_df=test_df.drop(['id', 'data source'], axis=1 )
train_path = '/data2/yinuo/covidx/train/'
test_path = '/data2/yinuo/covidx/test/'

train_df, valid_df = train_test_split(train_df, train_size=0.9, random_state=0)

class BaseImageDataset(Dataset):
    def __init__(self, split="train", transform=None) -> None:
        super().__init__()

        self.split = split
        self.transform = transform

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class COVIDXImageDataset(BaseImageDataset):
    def __init__(self, split="train", transform=None,
                 data_pct=0.01, imsize=256) -> None:
        super().__init__(split=split, transform=transform)

        self.train_df = train_df
        self.val_df = valid_df
        self.test_df = test_df

        self.imsize = imsize
        self.split = split

        if self.split == 'train':
            self.df = train_df
        elif self.split == 'val':
            self.df = valid_df
        else:
            self.df = test_df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        # get image
        if self.split == 'train':
            base_path = train_path
        elif self.split == 'val':
            base_path = train_path
        else:
            base_path = test_path
        img_path = base_path + row["filename"]

        x = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(x)

            
        y = row["class"]
        if y == 'positive':
            y = int(1)
        else:
            y = int(0)
        # y = torch.tensor([y])
        y = torch.tensor(y)

        return image, y