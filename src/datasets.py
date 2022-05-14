from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

od_transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4971, 0.4885, 0.4879], [0.2538, 0.2534, 0.2588])
])

od_transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4971, 0.4885, 0.4879], [0.2538, 0.2534, 0.2588])
])

ocr_transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.Normalize(0.3757, 0.4676)
])

ocr_transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(),
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
        super().__init__(root, od_transform_train if train else od_transform_val)


class ALPROCRDataset(_ALPRDataset):
    def __init__(self, root: str, train: bool) -> None:
        super().__init__(root, ocr_transform_train if train else ocr_transform_val)
