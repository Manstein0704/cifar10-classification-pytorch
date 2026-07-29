import torch
import torch.nn as nn
from torchvision import models


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


def create_resnet(n_output:int):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(
        model.fc.in_features,
        n_output
    )

    return model


def create_vgg16(n_output:int):
    model = models.vgg16_bn(weights=None)
    model.features[0] = nn.Conv2d(3, 64, 3, stride=1, padding=1)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, n_output, bias=True)

    return model

def get_model(model_name:str, n_output:int=10, n_hidden:int=128):
    model_name = model_name.lower()

    if model_name == "cnn":
        model = CNN(n_hidden, n_output)
        return model
    
    elif model_name == "resnet18":
        model = create_resnet(n_output)
        return model

    elif model_name == "vgg16":
        model = create_vgg16(n_output)
        return model

    else:
        raise ValueError(
            f"Unsupported model:{model_name}."
            "Choose from: cnn, resnet18, vgg16"
        )



if __name__ == "__main__":
    import torch
    x = torch.randn(2, 3, 32, 32)

    for model_name in ["cnn", "resnet18", "vgg16"]:
        model = get_model(model_name)
        model.eval()

        with torch.no_grad():
            output = model(x)

        print(f"{model_name}:"
              f"input={x.shape}"
              f"output={tuple(output.shape)}")