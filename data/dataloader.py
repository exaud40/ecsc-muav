# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 22:41:03 2025

@author: Exaud


"""
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except:
            return self.__getitem__((index+1) % len(self.samples))



def get_loader(path,batch=32):

    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])

    dataset = ImageFolder(path,transform)

    loader = DataLoader(dataset,
                        batch_size=batch,
                        shuffle=True,
                        num_workers=4)

    return loader


def get_loader_rs(path):

    tf = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    ds = SafeImageFolder(path,tf)

    tr = int(0.8*len(ds))
    te = len(ds)-tr

    tr_ds,te_ds = random_split(ds,[tr,te])

    return DataLoader(tr_ds,32,True), DataLoader(te_ds,32,False)