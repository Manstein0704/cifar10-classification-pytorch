from dataset import get_dataloaders
from trainer import train, evaluate
from models import get_model
from visualize import evaluate_history, plot_loss
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)


    train_loader, test_loader = get_dataloaders()

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    num_epochs = 50
    n_hidden = 128
    n_output = len(classes)
    model = get_model(args.model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    lr = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = np.zeros((0, 5))


    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"epoch:{epoch} train_loss:{train_loss} train_acc:{train_acc} test_loss:{test_loss} test_acc:{test_acc}")
        item = np.array([epoch+1, train_loss, train_acc, test_loss, test_acc])
        history = np.vstack((history, item))

    evaluate_history(history=history, model_name=args.model_name, save_path=f"outputs/{args.model_name}_loss.png")
    

    



def parser_args():
    parser = argparse.ArgumentParser(
        description="Train an image classification model on CIFAR-10."
    )
    parser.add_argument("--model_name",
                        type=str,
                        default="cnn",
                        choices=["cnn", "resnet18", "vgg16"],
                        help= "Model architecture to use. Default:cnn")

    return parser.parse_args()

if __name__ == "__main__":
    args = parser_args()
    main(args)

