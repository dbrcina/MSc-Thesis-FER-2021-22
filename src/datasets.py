from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

# TODO: Add augmentation?
TRAIN_TRANSFORM_OD = transforms.Compose([
    transforms.ToTensor(),
    # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
    # transforms.RandomAffine(degrees=50, translate=(0.1, 0.3), scale=(0.5, 0.75)),
    transforms.Normalize([0.4334, 0.4249, 0.4232], [0.2799, 0.2816, 0.2866])
])

VAL_TRANSFORM_OD = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.2799, 0.2816, 0.2866], [0.2799, 0.2816, 0.2866])
])


class ALPRODDataset(Dataset):
    def __init__(self, root: str, train: bool) -> None:
        transform = TRAIN_TRANSFORM_OD if train else VAL_TRANSFORM_OD
        self.dataset = datasets.ImageFolder(root, transform)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.dataset[idx]

    def __len__(self) -> int:
        return len(self.dataset)
