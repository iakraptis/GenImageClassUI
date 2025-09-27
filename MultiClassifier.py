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


