import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class ALPRDataset(Dataset):
    def __init__(self, root: str) -> None:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.dataset = datasets.ImageFolder(root, transform)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.dataset[idx]

    def __len__(self) -> int:
        return len(self.dataset)
