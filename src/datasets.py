import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

# TODO: add data augmentation
TRAIN_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x / 255.0),
    transforms.Normalize([0.42099491, 0.41037788, 0.4102286], [0.27918375, 0.28085432, 0.28571102])
])

VAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x / 255.0),
    transforms.Normalize([0.42099491, 0.41037788, 0.4102286], [0.27918375, 0.28085432, 0.28571102])
])


class ALPRDataset(Dataset):
    def __init__(self, root: str, train: bool) -> None:
        transform = TRAIN_TRANSFORM if train else VAL_TRANSFORM
        self.dataset = datasets.ImageFolder(root, transform)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.dataset[idx]

    def __len__(self) -> int:
        return len(self.dataset)
