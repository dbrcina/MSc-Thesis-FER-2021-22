from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

TRAIN_TRANSFORM_OD = transforms.Compose([
    transforms.ToTensor(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
    transforms.RandomAffine(degrees=50, translate=(0.1, 0.3), scale=(0.5, 0.75)),
    transforms.Normalize([0.42099491, 0.41037788, 0.4102286], [0.27918375, 0.28085432, 0.28571102])
])

VAL_TRANSFORM_OD = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.42099491, 0.41037788, 0.4102286], [0.27918375, 0.28085432, 0.28571102])
])


class ALPRODDataset(Dataset):
    def __init__(self, root: str, train: bool) -> None:
        transform = TRAIN_TRANSFORM_OD if train else VAL_TRANSFORM_OD
        self.dataset = datasets.ImageFolder(root, transform)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.dataset[idx]

    def __len__(self) -> int:
        return len(self.dataset)
