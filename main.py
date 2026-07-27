from dataset import get_dataloaders
from trainer import train, evaluate
from models import CNN
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)


    train_loader, test_loader = get_dataloaders()

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    num_epochs = 50
    n_hidden = 128
    n_output = len(classes)
    model = CNN(n_hidden, n_output)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr = 0.0001
    history = np.zeros((0, 5))


    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"epoch:{epoch} train_loss:{train_loss} train_acc:{train_acc} test_loss:{test_loss} test_acc:{test_acc}")
        item = np.array([epoch+1, train_loss, train_acc, test_loss, test_acc])
        history = np.vstack((history, item))



if __name__ == "__main__":
    main()

