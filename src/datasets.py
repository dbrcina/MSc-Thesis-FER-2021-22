from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

# TODO: Add augmentation?
TRAIN_TRANSFORM_OD = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4334, 0.4249, 0.4232], [0.2799, 0.2816, 0.2866])
])

VAL_TRANSFORM_OD = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.2799, 0.2816, 0.2866], [0.2799, 0.2816, 0.2866])
])

TRAIN_TRANSFORM_OCR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.Normalize(0.3757, 0.4676)
])

VAL_TRANSFORM_OCR = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(0.3757, 0.4676)
])


class _ALPRDataset(Dataset):
    def __init__(self, root: str, transform) -> None:
        self.dataset = datasets.ImageFolder(root, transform)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.dataset[idx]

    def __len__(self) -> int:
        return len(self.dataset)


class ALPRODDataset(_ALPRDataset):
    def __init__(self, root: str, train: bool) -> None:
        super().__init__(root, TRAIN_TRANSFORM_OD if train else VAL_TRANSFORM_OD)


class ALPROCRDataset(_ALPRDataset):
    def __init__(self, root: str, train: bool) -> None:
        super().__init__(root, TRAIN_TRANSFORM_OCR if train else VAL_TRANSFORM_OCR)
