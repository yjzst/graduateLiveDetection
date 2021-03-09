from PIL import Image
from torch.utils.data import Dataset,DataLoader
import os
import numpy as np
import pandas as pd
import torch
from torchvision import transforms

"""
author:YJZ
date:2020.5.19
"""


class CQNU_LN(Dataset):
    # 初始化构造函数
    def __init__(self,train_file_RGB=None,val_file_RGB=None,is_train=False,transform=None):
        self.train_file_RGB = train_file_RGB
        self.val_file_RGB = val_file_RGB
        self.transform = transform
        self.name_list = []
        self.is_train = is_train
        if is_train:
            if not os.path.isfile(self.train_file_RGB):
                print(self.train_file_RGB  + 'does not exist!')
            self.train_file_RGB = pd.read_csv(train_file_RGB)
            self.label_train = np.asanyarray(self.train_file_RGB.iloc[:,1])
            self.image_train_RGB = self.train_file_RGB.iloc[:,0]
            self.size_train = len(self.label_train)
        else:
            if not os.path.isfile(self.val_file_RGB):
                print(self.val_file_RGB  + 'does not exist!')
            self.val_file_RGB = pd.read_csv(val_file_RGB)
            self.image_val_RGB = self.val_file_RGB.iloc[:,0]
            self.label_val = np.asanyarray(self.val_file_RGB.iloc[:,1])
            self.size_val = len(self.label_val)
    def __len__(self):
        # 数据集的长度
        if self.is_train:
            return self.size_train
        else:
            return self.size_val
    def __getitem__(self, item):
        # 迭代数据集
        if self.is_train:
            RGB_train_path = self.image_train_RGB[item]
            if not os.path.isfile(RGB_train_path):
                print(RGB_train_path + 'does not exist')
                return None
            with Image.open(RGB_train_path) as img_RGB:
                Image_RGB_train = img_RGB.convert('RGB')
            label_train = torch.tensor(self.label_train[item],dtype=torch.long)
            if self.transform:
                Image_RGB_train = self.transform(Image_RGB_train)
            return Image_RGB_train,label_train
        else:
            RGB_val_path = self.image_val_RGB[item]
            if not os.path.isfile(RGB_val_path):
                print(RGB_val_path + 'or' + 'does not exist')
                return None
            with Image.open(RGB_val_path) as img_RGB:
                Image_RGB_val = img_RGB.convert('RGB')
            label_val = torch.tensor(self.label_val[item], dtype=torch.long)
            if self.transform:
                Image_RGB_val = self.transform(Image_RGB_val)
            return Image_RGB_val,label_val

if __name__ == '__main__':
    trans = transforms.Compose(transforms=[transforms.Resize(size=(112,112)),
                                           transforms.RandomResizedCrop(112),
                                           transforms.RandomHorizontalFlip(p=0.3),
                                           transforms.ToTensor()
                                           ])
    train_dataset = CQNU_LN(train_file_RGB=r"D:\liveness_detection\graduateLiveDetection\data\429Data.csv",
                            is_train=True,transform=trans)
    train_loader = DataLoader(dataset=train_dataset,batch_size=64,shuffle=False,num_workers=0)
    for i,(image_RGB,labels) in enumerate(train_loader):
        continue
    print("finish")
