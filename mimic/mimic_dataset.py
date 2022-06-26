import os
import json
import torch
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Dataset, default_collate, Subset
from torch.utils.data.sampler import SubsetRandomSampler
import random
import pandas as pd
import csv


class MimicDataset(Dataset):
    def __init__(self, split,
                 data_root='./mimic/',
                 ann_file='annotations_labels.json',
                 image_root='/data1/data/mimic-cxr/mimic-images-512/',
                 **kwargs):
        super(MimicDataset, self).__init__()
        assert split in ['train', 'validate', 'test']

        random.seed(1000)

        self.split = split
        self.transform = MimicDataset.get_transforms(split)
        self.image_root = image_root

        # new_dataroot = data_root + split + '.csv'
        # data = pd.read_csv(new_dataroot)
        # self.data = data
        # print('haha')

        # split the whole dataset into 8-1-1
        self.ann = json.loads(open(os.path.join(data_root, ann_file), 'r').read())
        a = self.ann
        self.total = a['train'] + a['validate'] + a['test']    # 377095
        # if split == 'train':

        # self.examples = self.ann[split]
        self.examples = self.total

        examples = []
        print("Processing labels")
        for example in tqdm(self.examples):
            try:
                image = Image.open(os.path.join(self.image_root, example['image_path']))
                label = np.array(example['label']).astype(np.float)
                label[label < 0] = 0.0
                # If no label is specified, skip (20249 examples)
                if sum(label) == 0:
                    continue
                example['label'] = label
                examples.append(example)

            except FileNotFoundError:
                # print('image not found for key', example['image_path'])
                # raise
                continue
                # pass

        # self.examples = examples    # 356225

        whole_len = len(examples)
        indices = list(range(whole_len))
        random.shuffle(indices)

        train_len = int(np.floor(0.8 * whole_len))   # 284980
        val_len = int(np.floor(0.1 * whole_len))    # 35622
        test_len = whole_len - train_len - val_len    # 35623
        train_idx = indices[:train_len]
        val_idx = indices[train_len : train_len + val_len]
        test_idx = indices[train_len + val_len :]

        self.trainset = Subset(examples, train_idx)
        self.valset = Subset(examples, val_idx)
        self.testset = Subset(examples, test_idx)
        
        if split == 'train':
            self.examples = self.trainset
        if split == 'validate':
            self.examples = self.valset
        if split == 'test':
            self.examples = self.testset


        # self.exp811 = {}
        # self.exp811['train'] = trainset
        # self.exp811['validate'] = valset
        # self.exp811['test'] = testset

        # pd.DataFrame(testset).to_csv('mimic/mine811/test.csv', quoting=csv.QUOTE_ALL)
        # pd.DataFrame(trainset).to_csv('mimic/mine811/train.csv', quoting=csv.QUOTE_ALL)
        # pd.DataFrame(valset).to_csv('mimic/mine811/validate.csv', quoting=csv.QUOTE_ALL)
        

        # pd.DataFrame(testset).to_json('mimic/mine811/test.json')
        # pd.DataFrame(trainset).to_json('mimic/mine811/train.json')
        # pd.DataFrame(valset).to_json('mimic/mine811/validate.json')

        # with open('mimic/mine811/try.json', 'w') as fp:
        #     json.dump(self.ann, fp)
    

    def __len__(self):
        return len(self.examples)
        # return len(self.data)

    def __getitem__(self, idx):
        example = self.examples[idx]
        # example = self.data.iloc[idx]
        study_id = example['study_id']
        subject_id = example['subject_id']
        image_path = example['image_path']

        key = (subject_id, study_id, image_path)


        try:
            image = self.transform(Image.open(os.path.join(self.image_root, image_path)).convert('RGB'))
            # image = Image.open(os.path.join(self.image_root, image_path)).convert('RGB')
        except FileNotFoundError:
            print('image not found for key', image_path)
            # raise
            return

        label = example['label']

        # return {'idx': idx,
        #         'key': key,
        #         'img': image,
        #         'label': label}
        return image, label

    @staticmethod
    def get_transforms(name):
        if name == 'train':
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.Resize((256, 256)),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

    @staticmethod
    def get_classes():
        return ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
                'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
                'Pneumothorax', 'Support Devices']

def my_collate(batch):
    batch = filter(lambda x: x is not None, batch)
    return default_collate(batch)


if __name__ == '__main__':
    # d = MimicDataset("train")
    # l = DataLoader(d, batch_size=16, shuffle=True, collate_fn=my_collate)
    # print(len(d))
    trainset = MimicDataset('train')
    valset = MimicDataset('validate')
    testset = MimicDataset('test')
    print(len(trainset))
    print(len(valset))
    print(len(testset))
