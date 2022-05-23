import string
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from mappings import text2labels

recognition_transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.Normalize(0.593, 0.2368)
])

recognition_transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.Normalize(0.593, 0.2368)
])

detection_transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4971, 0.4885, 0.4879], [0.2538, 0.2534, 0.2588])
])

detection_transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4971, 0.4885, 0.4879], [0.2538, 0.2534, 0.2588])
])


class _ALPRDataset(Dataset):
    def __init__(self, root: str, transform) -> None:
        self.dataset = ImageFolder(root, transform)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.dataset[idx]

    def __len__(self) -> int:
        return len(self.dataset)


class RecognitionDataset(_ALPRDataset):
    CHARS = list(string.digits + string.ascii_uppercase)
    CHAR2LABEL = {c: i + 1 for i, c in enumerate(CHARS)}
    LABEL2CHAR = {label: c for c, label in CHAR2LABEL.items()}

    def __init__(self, root: str, train: bool) -> None:
        super().__init__(root, recognition_transform_train if train else recognition_transform_val)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, lp_idx = super().__getitem__(idx)

        lp = self.dataset.classes[lp_idx]
        target = torch.tensor(text2labels(lp))
        target_length = torch.tensor([len(target)])

        return x, target, target_length

    @staticmethod
    def collate_fn(batch):
        x, targets, target_lengths = zip(*batch)
        x = torch.stack(x, dim=0)
        targets = torch.cat(targets, dim=0)
        target_lengths = torch.cat(target_lengths, dim=0)
        return x, targets, target_lengths


class DetectionDataset(_ALPRDataset):
    def __init__(self, root: str, train: bool) -> None:
        super().__init__(root, detection_transform_train if train else detection_transform_val)
