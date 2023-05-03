# 
# This file is part of the fedstellar framework (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2022 Enrique Tomás Martínez Beltrán.
#
import os
import sys
from math import floor

# To Avoid Crashes with a lot of nodes
import torch.multiprocessing
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10

torch.multiprocessing.set_sharing_strategy("file_system")


#######################################
#     CIFAR10DataModule for MNIST     #
#######################################


class CIFAR10DataModule(LightningDataModule):
    """
    LightningDataModule of partitioned MNIST.

    Args:
        sub_id: Subset id of partition. (0 <= sub_id < number_sub)
        number_sub: Number of subsets.
        batch_size: The batch size of the data.
        num_workers: The number of workers of the data.
        val_percent: The percentage of the validation set.
    """

    # Singleton
    cifar10_train = None
    cifar10_val = None

    def __init__(
            self,
            sub_id=0,
            number_sub=1,
            batch_size=32,
            num_workers=4,
            val_percent=0.1,
            iid=True,
    ):
        super().__init__()
        self.sub_id = sub_id
        self.number_sub = number_sub
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_percent = val_percent

        # Singletons of MNIST train and test datasets
        if not os.path.exists(f"{sys.path[0]}/data"):
            os.makedirs(f"{sys.path[0]}/data")

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor()
        ])

        if CIFAR10DataModule.cifar10_train is None:
            CIFAR10DataModule.cifar10_train = CIFAR10(
                f"{sys.path[0]}/data", train=True, download=True, transform=train_transform
            )
            if not iid:
                sorted_indexes = CIFAR10DataModule.cifar10_train.targets.sort()[1]
                CIFAR10DataModule.cifar10_train.targets = (
                    CIFAR10DataModule.cifar10_train.targets[sorted_indexes]
                )
                CIFAR10DataModule.cifar10_train.data = CIFAR10DataModule.cifar10_train.data[
                    sorted_indexes
                ]
        if CIFAR10DataModule.cifar10_val is None:
            CIFAR10DataModule.cifar10_val = CIFAR10(
                f"{sys.path[0]}/data", train=False, download=True, transform=transforms.ToTensor()
            )
            if not iid:
                sorted_indexes = CIFAR10DataModule.cifar10_val.targets.sort()[1]
                CIFAR10DataModule.cifar10_val.targets = CIFAR10DataModule.cifar10_val.targets[
                    sorted_indexes
                ]
                CIFAR10DataModule.cifar10_val.data = CIFAR10DataModule.cifar10_val.data[
                    sorted_indexes
                ]
        if self.sub_id + 1 > self.number_sub:
            raise ("Not exist the subset {}".format(self.sub_id))

        # Training / validation set
        trainset = CIFAR10DataModule.cifar10_train
        rows_by_sub = floor(len(trainset) / self.number_sub)
        tr_subset = Subset(
            trainset, range(self.sub_id * rows_by_sub, (self.sub_id + 1) * rows_by_sub)
        )
        cifar10_train, cifar10_val = random_split(
            tr_subset,
            [
                round(len(tr_subset) * (1 - self.val_percent)),
                round(len(tr_subset) * self.val_percent),
            ],
        )

        # Test set
        testset = CIFAR10DataModule.cifar10_val
        rows_by_sub = floor(len(testset) / self.number_sub)
        te_subset = Subset(
            testset, range(self.sub_id * rows_by_sub, (self.sub_id + 1) * rows_by_sub)
        )

        if len(testset) < self.number_sub:
            raise ("Too much partitions")

        # DataLoaders
        self.train_loader = DataLoader(
            cifar10_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            cifar10_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            te_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        print(
            "Train: {} Val:{} Test:{}".format(
                len(cifar10_train), len(cifar10_val), len(te_subset)
            )
        )

    def train_dataloader(self):
        """ """
        return self.train_loader

    def val_dataloader(self):
        """ """
        return self.val_loader

    def test_dataloader(self):
        """ """
        return self.test_loader


if __name__ == "__main__":
    dm = CIFAR10DataModule()
    print(dm.train_dataloader())
    print(dm.val_dataloader())
    print(dm.test_dataloader())
