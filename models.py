import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, n_hidden, n_output):
        super().__init__()
        self.Conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.Conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.Conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.Conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.Conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.Conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d((2, 2))
        self.flatten = nn.Flatten()
        self.l1 = nn.LazyLinear(n_hidden)
        self.l2 = nn.Linear(n_hidden, n_output)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.4)
        self.dropout4 = nn.Dropout(0.5)

        self.features = nn.Sequential(
            self.Conv1,
            self.relu,
            self.Conv2,
            self.relu,
            self.maxpool,
            self.dropout1,
            self.Conv3,
            self.relu,
            self.Conv4,
            self.relu,
            self.maxpool,
            self.dropout2,
            self.Conv5,
            self.relu,
            self.Conv6,
            self.relu,
            self.maxpool,
            self.dropout3
        )

        self.classifier = nn.Sequential(
            self.flatten,
            self.l1,
            self.relu,
            self.dropout4,
            self.l2,
        )

    def forward(self, x):
        x1 = self.features(x)
        x2 = self.classifier(x1)
        return x2


