import torch
from tqdm import tqdm

def train(model, train_loader, criterion, optimizer, device):
    model.train()

    n_train = 0
    n_correct = 0
    total_loss = 0

    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * (len(labels))
        predicitions = outputs.argmax(dim=1)
        n_correct += (predicitions==labels).sum().item()
        n_train += len(labels)

    train_acc = n_correct / n_train
    train_loss = total_loss / n_train
    return train_loss, train_acc


@torch.no_grad()
def evaluate(model, test_loader, criterion, device):
    model.eval()

    n_val = 0
    n_correct = 0
    total_loss = 0

    for images, labels in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * len(labels)
        predicted = outputs.argmax(dim=1)
        n_correct += (predicted==labels).sum().item()
        n_val += len(labels)

    test_acc = n_correct / n_val
    test_loss = total_loss / n_val

    return test_loss, test_acc

        
        
