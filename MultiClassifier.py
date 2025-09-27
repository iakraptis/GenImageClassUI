import pathlib
from random import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.nn import (
    BCELoss,
    CrossEntropyLoss,
    Linear,
    Module,
    ReLU,
    Sigmoid,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.models import ResNet50_Weights, resnet50, vgg16
from torchvision.utils import make_grid



class FakeImageClassifier(Module):
    def __init__(self):
        super(FakeImageClassifier, self).__init__()

        # Load the VGG16 model and remove its final classification layer
        # self.vgg = vgg16(pretrained=True)
        # self.vgg.classifier = Sequential(*list(self.vgg.classifier.children())[:-1])
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.resnet.fc = Linear(2048, 1024)
        # # freeze the VGG layers
        # for param in self.vgg.parameters():
        #     param.requires_grad = False

        # Additional layers for multi classification
        self.fc1 = Linear(1024, 512)
        self.fc2 = Linear(512, 128)
        self.fc3 = Linear(128, 4)

        self.relu = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, x):
        # Pass input through VGG layers
        # x = self.vgg(x)
        x = self.resnet(x)

        # Pass through custom layers for binary classification
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def train_model(
    model: Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    writer: SummaryWriter,
    epochs: int = 10,
):
    criterion = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # Try to add graph to tensorboard if possible; guard in case of shape/device issues
    try:
        sample_inputs = next(iter(train_loader))[0]
        writer.add_graph(model, sample_inputs.to(next(model.parameters()).device))
    except Exception:
        # Skip graph logging if it fails (common with complex models/transforms)
        pass
    # Ensure model is on the same device as inputs/parameters
    model.to(next(model.parameters()).device)
    writer.add_text("Model", str(model))
    max_accuracy = 0
    no_better_model_patience = 0
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device).long()
            optimizer.zero_grad()
            predictions = model(inputs)
            #breakpoint()
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            predicted = torch.argmax(predictions, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            batch_accuracy = 100 * correct / total
            loss.backward()
            optimizer.step()
            print(
                f"Epoch {epoch + 1}, Batch {i + 1}, Batch Accuracy: {batch_accuracy:.2f}%"
            )
        train_accuracy = 100 * correct / total
        print(f"===> Epoch {epoch + 1}, Accuracy: {train_accuracy:.2f}%")
        writer.add_scalar("Accuracy/Train", train_accuracy, epoch)

        # Test the model on the validation set
        test_accuracy = test_model(model, validation_loader)
        writer.add_scalar("Accuracy/Test", test_accuracy, epoch)

        no_better_model_patience += 1

        # If we have better model save it
        if max_accuracy <= test_accuracy:
            max_accuracy = test_accuracy
            no_better_model_patience = 0
            print("Better Model Achieved, Saving Model")
            save_model(model, "triplemodel2.pth")

        # Early stop if there is no improvement for training
        if no_better_model_patience >= 3:
            print(f"Best Model Accuracy: {max_accuracy:.2f}%")
            print("Early Stopping")
            break

    print("Finished Training")


def show_images(images, labels, predictions):
    # Ensure that batch_size is not larger than the grid capacity (3x3)
    batch_size = min(images.size(0), 9)  # Limit to 9 images for 3x3 grid

    # Create a 3x3 grid
    fig, ax = plt.subplots(3, 3, figsize=(12, 12))
    fig.tight_layout(pad=3)

    for i in range(batch_size):
        row = i // 3
        col = i % 3
        # Move image to cpu and convert to numpy for matplotlib
        img = images[i].detach().cpu()
        # Unnormalize assuming Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        img = img * 0.5 + 0.5
        img_np = np.clip(np.transpose(img.numpy(), (1, 2, 0)), 0, 1)
        ax[row, col].imshow(img_np)
        # Get predicted class index
        pred_class = int(torch.argmax(predictions[i]).item())
        ax[row, col].set_title(f"Label: {labels[i]}, Pred: {pred_class}")
        ax[row, col].axis("off")

    # Hide any unused subplots if batch_size < 9
    for i in range(batch_size, 9):
        fig.delaxes(ax[i // 3, i % 3])

    # plt.show()

    return fig


def test_model(model, test_loader):
    correct = 0
    total = 0
    # Preserve original device of model parameters if set
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device).long()
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")
    model.train()  # Set the model back to training mode if needed
    return accuracy



def save_model(model, path):
    torch.save(model.state_dict(), path)


# Use 3-channel mean/std for RGB images
train_transform = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.RandomResizedCrop(512),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomPerspective(),
        transforms.RandomRotation(45),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
inference_transform = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


if __name__ == "__main__":
    # Initialize model
    model = FakeImageClassifier()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize Train & Test Data
    data_dir = pathlib.Path(__file__).parent.parent.absolute() / "Dataset" / "Images"
    train_dir = data_dir / "Train"
    validation_dir = data_dir / "Valid"
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    # print the classes
    print(train_dataset.classes)
    validation_dataset = datasets.ImageFolder(
        root=validation_dir, transform=inference_transform
    )
    validation_loader = DataLoader(validation_dataset, batch_size=20, shuffle=True)
    
    # Initialize Tensorboard
    writer = SummaryWriter()

    # Train Model
    train_model(model, train_loader, validation_loader, writer, epochs=40)

    # Save Model
    save_model(model, "quadmodel.pth")

    # # Load Model
    # model.load_state_dict(torch.load("quadmodel.pth"))
    # # test model
    # test_model(model, validation_loader)
