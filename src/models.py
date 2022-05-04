import torch
from torch import nn


class LeNet5(nn.Module):
    def __init__(self, n_classes: int, dropout: float = 0.3) -> None:
        super().__init__()

        self.n_classes = n_classes
        self.dropout = dropout

        act_fun = nn.ReLU(inplace=True)
        drop_fun = nn.Dropout(p=dropout)

        # In the paper, 32x32 image is expected, but here 28x28 is given, so padding is set to 2
        self.features = nn.Sequential(
            # 1
            nn.Conv2d(1, 6, kernel_size=5, padding=2),  # Nx6x28x28
            act_fun,
            nn.MaxPool2d(kernel_size=2, stride=2),  # Nx6x14x14
            # 2
            nn.Conv2d(6, 16, kernel_size=5),  # Nx16x10x10
            act_fun,
            nn.MaxPool2d(kernel_size=2, stride=2),  # Nx16x5x5
            # 3
            nn.Conv2d(16, 120, kernel_size=5),  # Nx120x1x1
            act_fun,
            nn.Flatten()  # Nx120 (120*1*1)
        )
        self.classifier = nn.Sequential(
            # 4
            nn.Linear(120, 84),  # Nx84
            act_fun,
            # 5
            drop_fun,
            nn.Linear(84, n_classes)  # Nxn_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1:] == (1, 28, 28), "Expecting a grayscale image with size 28x28"

        x = self.features(x)
        x = self.classifier(x)

        return x


class AlexNet(nn.Module):
    def __init__(self, n_classes: int, dropout: float = 0.5) -> None:
        super().__init__()

        self.n_classes = n_classes
        self.dropout = dropout

        act_fun = nn.ReLU(inplace=True)
        drop_fun = nn.Dropout(p=dropout)

        self.features = nn.Sequential(
            # 1
            nn.Conv2d(3, 32, kernel_size=11, stride=4),  # Nx32x55x55
            act_fun,
            nn.MaxPool2d(kernel_size=3, stride=2),  # Nx32x27x27
            # 2
            nn.Conv2d(32, 64, kernel_size=5, padding=2),  # Nx64x27x27
            act_fun,
            nn.MaxPool2d(kernel_size=3, stride=2),  # Nx64x13x13
            # 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Nx128x13x13
            act_fun,
            # 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # Nx128x13x13
            act_fun,
            # 5
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # Nx64x13x13
            act_fun,
            nn.MaxPool2d(kernel_size=3, stride=2),  # Nx64x6x6
            nn.Flatten()  # Nx2304 (64*6*6)
        )
        self.classifier = nn.Sequential(
            # 6
            drop_fun,
            nn.Linear(2304, 1024),  # Nx1024
            act_fun,
            # 7
            drop_fun,
            nn.Linear(1024, 256),  # Nx256
            act_fun,
            # 8
            nn.Linear(256, n_classes)  # Nxn_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1:] == (3, 227, 227), "Expecting a color image with size 227x227"

        x = self.features(x)
        x = self.classifier(x)

        return x
