from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

import config
from mappings import text2labels

recognition_transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomPerspective(distortion_scale=config.LP_DISTORTION),
    transforms.RandomRotation(config.LP_ROT_ANGLE, interpolation=transforms.InterpolationMode.BILINEAR),
])

recognition_transform_val = transforms.Compose([
    transforms.ToTensor(),
])

detection_transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomPerspective(distortion_scale=config.LP_DISTORTION),
    transforms.RandomRotation(config.LP_ROT_ANGLE, interpolation=transforms.InterpolationMode.BILINEAR),
])

detection_transform_val = transforms.Compose([
    transforms.ToTensor(),
])


# Default pil_loader in ImageFolder implementation forces RGB conversion.
# This is not needed because dataset will always be grayscale.
def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("L")


class _ALPRDataset(Dataset):
    def __init__(self, root: str, transform) -> None:
        self.dataset = ImageFolder(root, transform, loader=pil_loader)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.dataset[idx]

    def __len__(self) -> int:
        return len(self.dataset)


class RecognitionDataset(_ALPRDataset):
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
