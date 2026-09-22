'''
Using the PathMNIST dataset to build and evaluate deep-learning models that classify colorectal histology
image patches into one of nine tissue classes. PathMNIST provides RGB images of size 28 x 28 pixels, with
training, validation, and test splits (20,000/3,000/3,000 images). Use these supplied splits; do not
merge or resplit them. You can use data = np.load("train_20000.npz") to load the images. You need to
construct a deep CNN of your choice to maximize the performance. However, it is essential that there
are at least two CONV layers. Print the loss and training accuracy periodically after each epoch. You
need to train the model for at least 3 epochs. Use the trained model to compute the test accuracy across
the test set and print the test accuracy.
'''
# Tasks:
#
#     Loading the dataset (5 Marks)
#
#     Preprocess the data (10 Marks)
#
#     Build a deep neural network & report the results (20 Marks)


# plan
# input --> conv --> BN --> ReLU --> Conv --> BN --> ReLU --> MaxPool
# --> MaxPool --> Dropout --> Conv --> BN --> ReLU --> Conv --> BN --> ReLU
# --> MaxPool --> Dropout --> FC --> FC(9)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Load data
train_data = np.load("data_q2/train_20000.npz")
val_data = np.load("data_q2/val_3000.npz")
test_data = np.load("data_q2/test_3000.npz")

print("Train keys: ", train_data.files)
print("Validation keys: ", val_data.files)
print("Test keys: ", test_data.files)
# this tells that its a dictionary type of data
# Train keys:  ['images', 'labels']
# Validation keys:  ['images', 'labels']
# Test keys:  ['images', 'labels']

# extract images and labels
x_train = train_data["images"]
y_train = train_data["labels"]

x_val = val_data["images"]
y_val = val_data["labels"]

x_test = test_data["images"]
y_test = test_data["labels"]

print("Training :", x_train.shape, y_train.shape)
print("Validation :", x_val.shape, y_val.shape)
print("Test :", x_test.shape, y_test.shape)


# Preprocess data
# convert pixel values from [0,255] to [0,1]
x_train = x_train.astype(np.float32) / 255
x_val = x_val.astype(np.float32) / 255
x_test = x_test.astype(np.float32) / 255

# convert images from Numpy format:
# (N,64,64,3) (N, Height, Width, Channels)
# to pytorch CNN format:
# (N,3,64,64) (N, Channels, Height, Width)

x_train = np.transpose(x_train, (0, 3, 1, 2))
x_val = np.transpose(x_val, (0, 3, 1, 2))
x_test = np.transpose(x_test, (0, 3, 1, 2))

# Flatten labels
y_train = y_train.reshape(-1)
y_val = y_val.reshape(-1)
y_test = y_test.reshape(-1)

# convert numpy arrays to pytorch tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
x_val = torch.tensor(x_val, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

print("After preprocessing:")
print("Training images: ", x_train.shape)
print("Training labels: ", y_train.shape)
print("Validation images: ", x_val.shape)
print("Validation labels: ", y_val.shape)
print("Test images: ", x_test.shape)
print("Test labels: ", y_test.shape)

# create datasets and dataloaders
train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_val, y_val)
test_dataset = TensorDataset(x_test, y_test)

batch_size = 128

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# the CNN
class PathMNISTCNN(nn.Module):
    def __init__(self):
        super(PathMNISTCNN, self).__init__()

        self.features = nn.Sequential(
            # conv block 1
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),

            # conv block 2
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),

            # conv block 3
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        )

        # image size calc
        # input = 64 x 64
        # maxpool 1: 64 -> 32
        # maxpool 2: 32 -> 16
        # maxpool 3: 16 -> 8
        # final feature map: 128 x 8 x 8

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 9)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# select cpu or gpu
device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

print(device)

# model
model = PathMNISTCNN().to(device)
print(model)

# loss func and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

# train the model
num_epochs = 10

print("Training...")

for epoch in range(num_epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)

        # calculate loss
        loss = criterion(outputs, labels)

        # clear previous gradients
        optimizer.zero_grad()

        loss.backward()  # backprop
        optimizer.step()  # updating the weights

        # calc training statistics
        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    # avg loss
    epoch_loss = running_loss / total

    # training accuracy
    train_accuracy = 100 * correct / total

    # validation
    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

    val_accuracy = 100 * val_correct / val_total

    # print results for this epoch
    print(
        f"Epoch [{epoch + 1}/{num_epochs}] | "
        f"Loss: {epoch_loss:.4f} | "
        f"Training Accuracy: {train_accuracy:.2f}% | "
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )


# evaluate the model on test data
model.eval()

test_total = 0
test_correct = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        test_total += labels.size(0)

        test_correct += (
            predicted == labels
        ).sum().item()


test_accuracy = 100.0 * test_correct / test_total

print("Final Results")

# final results
print(f"Test Accuracy: {test_accuracy:.2f}%")

print("=" * 40)