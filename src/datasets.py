import torch
from torch.utils import data
from torchvision import datasets, transforms


class ALPRDataset(data.Dataset):
    def __init__(self, root: str) -> None:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.dataset = datasets.ImageFolder(root, transform)
        self.dataset.

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.dataset[idx]

    def __len__(self) -> int:
        return len(self.dataset)


if __name__ == "__main__":
    dataset = ALPRDataset("../data")
    train_size = int(0.8*len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])
    dataloader = data.DataLoader(dataset, batch_size=2)
    # for batch in dataloader:
    #     x, y = batch
    #     print(batch)
    #     break
