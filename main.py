from dataset import get_dataloaders

def main():
    train_loader, test_loader = get_dataloaders()
    for image, label in train_loader:
        break
    print(image.shape)
    print(label.shape)



if __name__ == "__main__":
    main()

