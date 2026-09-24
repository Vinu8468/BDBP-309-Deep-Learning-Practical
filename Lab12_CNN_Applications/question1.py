# Download MNIST dataset and implement a MNIST classifier
# using CNN PyTorch library.

import torch
import torch.nn as nn
import torch.optim as optim
from click.core import F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#Transform
transform = transforms.ToTensor()

# Download MNIST dataset
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True
)
# Data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

# CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3
        )
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        self.fc1 = nn.Linear(
            in_features=64*5*5,
            out_features=128
        )
        self.fc2 = nn.Linear(
            in_features=128,
            out_features=10
        )
    def forward(self, x):
        # First conv
        x = self.conv1(x)
        # then ReLU
        x = torch.relu(x)
        # then max pool
        x = self.pool(x)
        # second conv
        x = self.conv2(x)
        # ReLU
        x = torch.relu(x)
        # Max pool
        x = self.pool(x)
        # flatten
        x = x.view(x.size(0), -1)
        # FC
        x = self.fc1(x)
        # ReLU
        x = torch.relu(x)
        #output
        x = self.fc2(x)
        return x
# create model
model = CNN()
#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        # forword pass
        outputs = model(images)
        # calculate the loss
        loss = criterion(outputs, labels)
        # clear previous gradients
        optimizer.zero_grad()
        # backprop
        loss.backward()
        #update weights
        optimizer.step()
        running_loss += loss.item()

    print(
        "Epoch:",epoch+1,
        "Loss:",running_loss/len(train_loader)
    )

# testing
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct +=(predicted ==labels).sum().item()
accuracy = 100*correct/total
print("Accuracy:", accuracy,"%")
