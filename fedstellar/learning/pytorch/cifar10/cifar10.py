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


class LitCIFAR10DataModule(pl.LightningDataModule):
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
            # dataset = CIFAR10DatasetCustom(
            #     cfg=cfg,
            #     train=train,
            #     transform=transform,
            # )
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
            pin_memory=True,
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
            pin_memory=True,
        )

        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()


class CIFAR10DatasetCustom(torch.utils.data.Dataset):
    def __init__(self, cfg, train, transform=None):
        super(CIFAR10DatasetCustom, self).__init__()
        self.transform = transform
        self.cfg = cfg
        self.split_dir = "train" if train else "test"
        self.root_dir = Path(cfg.dataset.root_dir)
        self.image_dir = self.root_dir / "cifar" / self.split_dir
        self.file_list = [p.name for p in self.image_dir.rglob("*") if p.is_file()]
        self.labels = [re.split("_|\.", l)[1] for l in self.file_list]
        self.targets = self.label_mapping(cfg)

    def label_mapping(self, cfg):
        labels = self.labels
        label_mapping_path = Path(cfg.dataset.root_dir) / "cifar/labels.txt"
        df_label_mapping = pd.read_table(label_mapping_path.as_posix(), names=["label"])
        df_label_mapping["target"] = range(cfg.train.num_classes)

        label_mapping_dict = dict(
            zip(
                df_label_mapping["label"].values.tolist(),
                df_label_mapping["target"].values.tolist(),
            )
        )

        targets = [label_mapping_dict[i] for i in labels]
        return targets

    def __getitem__(self, index):
        filename = self.file_list[index]
        targets = self.targets[index]
        image_path = self.image_dir / filename
        image = Image.open(image_path.as_posix())

        if self.transform is not None:
            transform = self.transform
            image = transform(image)

        return image, targets

    def __len__(self):
        return len(self)
