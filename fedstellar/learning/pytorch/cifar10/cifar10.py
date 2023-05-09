# 
# This file is part of the fedstellar framework (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2022 Enrique Tomás Martínez Beltrán.
#
from math import floor

# To Avoid Crashes with a lot of nodes
import torch.multiprocessing
import lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10

import re
from pathlib import Path
from PIL import Image
import pandas as pd


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, normalization="cifar10", loading="torchvision", sub_id=0, number_sub=1, num_workers=4, batch_size=32, iid=True, root_dir="./data"):
        super().__init__()
        self.sub_id = sub_id
        self.number_sub = number_sub
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.iid = iid
        self.root_dir = root_dir
        self.loading = loading
        self.normalization = normalization
        self.mean = self.set_normalization(normalization)["mean"]
        self.std = self.set_normalization(normalization)["std"]

    def set_normalization(self, normalization):
        # Image classification on the CIFAR10 dataset - Albumentations Documentation https://albumentations.ai/docs/autoalbument/examples/cifar10/
        if normalization == "cifar10":
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2471, 0.2435, 0.2616)
        elif normalization == "imagenet":
            # ImageNet - torchbench Docs https://paperswithcode.github.io/torchbench/imagenet/
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        else:
            raise NotImplementedError
        return {"mean": mean, "std": std}

    def get_dataset(self, train, transform, download=True):
        if self.loading == "torchvision":
            dataset = CIFAR10(
                root=self.root_dir,
                train=train,
                transform=transform,
                download=download,
            )
        elif self.loading == "custom":
            raise NotImplementedError
        else:
            raise NotImplementedError
        return dataset

    def train_dataloader(self):
        transform = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = self.get_dataset(
            train=True,
            transform=transform,
        )

        # To Avoid same data in all nodes
        rows_by_sub = floor(len(dataset) / self.number_sub)
        cifar10_train = torch.utils.data.Subset(dataset, range(self.sub_id * rows_by_sub, (self.sub_id + 1) * rows_by_sub))

        dataloader = DataLoader(
            cifar10_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=False,
        )

        print(f"Train Dataset Size: {len(cifar10_train)}")

        return dataloader

    def val_dataloader(self):
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = self.get_dataset(train=False, transform=transform)
        # To Avoid same data in all nodes
        rows_by_sub = floor(len(dataset) / self.number_sub)
        cifar10_val = torch.utils.data.Subset(dataset, range(self.sub_id * rows_by_sub, (self.sub_id + 1) * rows_by_sub))
        print(f"Val/Test Dataset Size: {len(cifar10_val)}")
        print(f"Example: {cifar10_val[0][0].shape}")
        dataloader = DataLoader(
            cifar10_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=False,
        )

        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()
