'''
Splice junctions are points on a DNA sequence at which 'superfluous' DNA is removed during
the process of protein creation in higher organisms. The problem posed in this dataset is to recognize,
given a sequence of DNA, the boundaries between exons (the parts of the DNA sequence retained after splicing)
and introns (the parts of the DNA sequence that are spliced out). This problem consists of two subtasks:
recognizing exon/intron boundaries (referred to as El sites), and recognizing intron/exon boundaries
(IE sites). (In the biological community, IE borders are referred to as "acceptors" while El borders are
referred to as "donors".)
'''
# Using the UCI Splice-junction Gene Sequences dataset, build and evaluate a FFN deep-learning model that
# classify each 60-base DNA sequence as:

# - El: exon-intron boundary
# - IE: intron-exon boundary
# - N: neither

'''The dataset contains 3,190 labelled sequences. Do not use the instance-name field as an input feature.
You can use one-hot representation to encode a base. Use train-test-split (80:20) with stratification
to form train and test sets. Use the seed value of 42 for train-test-split. You need to construct a 
feed forward deep neural network of your choice to maximize the performance. However, it is essential 
that there are at least two FC layers. Print the training loss at the end of each epoch. You need to
train for atleast 5 epochs. Print the overall accuracy and the macro F1-score on the test set using the
trained model.'''

# Tasks:
#
#     Loading the dataset (5 Marks)
#
#     Preprocess the sequences (10 Marks)
#
#     Build a deep neural network & report the results (20 Marks)

# import libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score

from torch.utils.data import TensorDataset, DataLoader

# load data
file_path = "data_q1/splice.data"

# read the file
sequences = []
labels = []

with open(file_path, "r") as file:
    for line in file:
        line = line.strip()
        # skip empty lines
        if not line:
            continue
        # each line contains:
        # class, sequence
        parts = line.split(",")
        label = parts[0].strip()
        instance_name = parts[1].strip()
        sequence = parts[2].strip()

        labels.append(label)
        sequences.append(sequence)

print(f"Number of sequences: {len(sequences)}")
print(f"Number of labels: {len(labels)}")

# check data
print("First 5 sequences: ")
for i in range(5):
    print(
        labels[i],
        sequences[i],
    )

print("Sequence lenth:",len(sequences[0]))

# preprocess the sequence

# one hot encode it

base_to_one_hot ={
    "A":[1,0,0,0],
    "C":[0,1,0,0],
    "G":[0,0,1,0],
    "T":[0,0,0,1],

    "N":[0,0,0,0],
    "D":[0,0,0,0],
    "R":[0,0,0,0],
    "S":[0,0,0,0]
}

def one_hot_encode(sequence):
    encoded_sequence = []
    for base in sequence:
        encoded_sequence.extend(
            base_to_one_hot[base]
        )
    return encoded_sequence

# encode all sequence
X = np.array(
    [one_hot_encode(sequence)
     for sequence in sequences]
    ,dtype=np.float32
)

print("Shape after one-hot encoding of sequences:")
print(X.shape)

# encode labels
#convert class names into integers

# EI --> 0 , IE --> 1 , N --> 2

label_to_number = {
    "EI" : 0,
    "IE" : 1,
    "N" : 2
}

y = np.array(
    [
        label_to_number[label]
        for label in labels
    ],
    dtype=np.int64
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
    random_state=42,stratify=y)

print("training :",X_train.shape,y_train.shape)
print("testing :",X_test.shape,y_test.shape)

# convert numpy arrays to pytorch tensors
X_train = torch.tensor(X_train,dtype=torch.float32)
X_test = torch.tensor(X_test,dtype=torch.float32)

# same with class labels also since CrossEntropyLoss requires
y_train = torch.tensor(y_train,dtype=torch.long)
y_test = torch.tensor(y_test,dtype=torch.long)

# create datasets and dataloaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Feed Forward Neural Network
class SpliceFNN(nn.Module):
    def __init__(self):
        super(SpliceFNN, self).__init__()
        self.network = nn.Sequential(
            # FC layer 1
            nn.Linear(in_features=240,out_features=256),
            nn.ReLU(),
            nn.Dropout(0.3),

            # FC layer 2
            nn.Linear(in_features=256,out_features=128),
            nn.ReLU(),
            nn.Dropout(0.3),

            # FC layer 3
            nn.Linear(in_features=128,out_features=64),
            nn.ReLU(),

            # Output layer
            nn.Linear(in_features=64,out_features=3)

        )
    def forward(self, x):
        x = self.network(x)
        return x

# select cpu or gpu(we know the answer its cpu)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# create model
model = SpliceFNN().to(device)
print(model)

# loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

# train the model
num_epochs = 20
print("Trainnnninnnggg.......")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    total = 0

    for sequences_batch,labels_batch in train_loader:
        sequences_batch = sequences_batch.to(device)
        labels_batch = labels_batch.to(device)

        # forward pass
        outputs = model(sequences_batch)

        # loss calc
        loss = criterion(outputs, labels_batch)

        # clear previous gradients
        optimizer.zero_grad()

        # backprop
        loss.backward()

        # update weights
        optimizer.step()

        # calc training loss
        running_loss += (
            loss.item()*sequences_batch.size(0)
        )
        total += sequences_batch.size(0)

    # avg training loss
    epoch_loss = running_loss/total

    #print training loss after every epoch
    print(
        f"Epoch [{epoch+1}/{num_epochs}] |",
        f"Training Loss: {epoch_loss:.3f}"
    )

# evaluate the model on test data
model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for sequences_batch,labels_batch in test_loader:
        sequences_batch = sequences_batch.to(device)
        labels_batch = labels_batch.to(device)

        # forward pass
        outputs = model(sequences_batch)

        # get predicted class
        _, predicted = torch.max(outputs, 1)

        # store predictions
        all_predictions.extend(
            predicted.cpu().numpy()
        )

        all_labels.extend(
            labels_batch.cpu().numpy()
        )

# calculate test accurary
test_accuracy = accuracy_score(all_labels, all_predictions)

# calculate f1 score
test_f1 = f1_score(all_labels, all_predictions,average="macro")

# final results
print("Final Results:")
print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Macro F1-score: {test_f1:.2f}%")

print("That's a wrap")