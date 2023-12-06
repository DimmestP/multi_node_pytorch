import os
import pandas as pd
import numpy
from torchvision.io import read_image
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Resize, Lambda

device = torch.device('cuda')

# Import data

class ImagenetDataset(Dataset):
	def __init__(self, annotations_file, img_dir):
		self.img_labels_raw = pd.read_csv(annotations_file, sep='\t', header = None)
		self.img_labels = torch.LongTensor(numpy.array(pd.get_dummies(self.img_labels_raw[1])))
		self.img_dir = img_dir
		self.transform = Resize([300,300], antialias=True)

	def __len__(self):
		return len(self.img_labels)

	def __getitem__(self, idx):
		img_path = os.path.join(self.img_dir, self.img_labels_raw.iloc[idx, 0])
		image = read_image(img_path).to(torch.float32)
		label = self.img_labels[idx]
		image = self.transform(image)
		return image, label

training_data = ImagenetDataset(annotations_file = "/mnt/ceph_rbd/imagenet_data/imagenet_data_subset/train/image_labels.txt",
				img_dir = "/mnt/ceph_rbd/imagenet_data/imagenet_data_subset/train/")

validation_data = ImagenetDataset(annotations_file = "/mnt/ceph_rbd/imagenet_data/imagenet_data_subset/validation/image_labels.txt",
                                img_dir = "/mnt/ceph_rbd/imagenet_data/imagenet_data_subset/validation/")

train_dataloader = DataLoader(training_data, batch_size=128, shuffle=True)

validation_dataloader = DataLoader(validation_data, batch_size=128, shuffle=True)

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(300*300*3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.MultiLabelMarginLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(validation_dataloader, model, loss_fn)

print("Done!")
