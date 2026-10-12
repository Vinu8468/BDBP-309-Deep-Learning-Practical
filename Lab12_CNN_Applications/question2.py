# Implement CNN using PyTorch for image classification
# using cifar10 dataset -
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html . Plot train error vs increasing number of layers. After some point,
# the training error increases with the number of layers.

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

# load the cifer-10 dataset
transform = transforms.ToTensor()

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Use a subset to keep the experiment simple and faster
trainset = Subset(trainset, range(10000))
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CNN with variable number of convolution layer.. new thing to try
class CNN(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        layers = []
        in_channels = 3
        for i in range(num_layers):
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
            in_channels = 32

            if (i+1) % 2 == 0:
                layers.append(nn.MaxPool2d(2)) #introducing pooling every othre layer
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(in_features=32, out_features=10)

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

#now trin the cnn with increasing depth
depths = [1,2,3,4,5]
train_errors = []

for depth in depths:
    model = CNN(depth).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(100):
        model.train()
        for images, labels in trainloader:
            images,labels = images.to(device),labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    # calculate training error
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in trainloader:
            images,labels = images.to(device),labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    error = 100 * (1-correct / total)
    train_errors.append(error)

    print(f"Layers: {depths}, Training Error: {error:.2f}%")

# ploting training error vs number of layers
plt.plot(depths, train_errors, marker='o')
plt.xlabel("Number of Conv layers")
plt.ylabel("Training Error")
plt.title("Training Error vs Number of Conv Layers")
plt.xticks(depths)
plt.grid(True)
plt.show()
