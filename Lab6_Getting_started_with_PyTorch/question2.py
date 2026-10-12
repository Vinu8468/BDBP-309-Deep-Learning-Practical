# Learn about Tensors, Datasets and Dataloaders,
# Transforms, BuildModel, Autograd, Optimization
# and Save & load model - Try the code snippets for
# each of the above operations from here

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset,TensorDataset
from torchvision import transforms
from PIL import Image
import os

# Tensors
print("Tensors")
x = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
print("Tensor:",x)
print("Shape:",x.shape)
print("Addition:",x+x)

# Datasets and Dataloaders
print("Dataset and loader")
features = torch.randn(100,3)
labels = torch.randint(0,2,(100,))
dataset = TensorDataset(features,labels)
loader = DataLoader(dataset,batch_size=10,shuffle=True)

for batch_x ,batch_y in loader:
    print("batch shape:",batch_x.shape)
    break

# Transforms
print("Transforms")
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
])
print("Image Transform defined:",transform)

# build a model
print("Model building")
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3,8),
            nn.ReLU(),
            nn.Linear(8,2)
        )
    def forward(self, x):
        return self.network(x)
model = SimpleModel()
print(model)

#still work remaining

