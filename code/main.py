import os
import pandas as pd
import numpy
from torchvision.io import read_image
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Resize, Lambda

class ImagenetDataset(Dataset):
	def __init__(self, annotations_file, img_dir):
		self.img_labels_raw = pd.read_csv(annotations_file, sep='\t')
		self.img_labels = torch.IntTensor(numpy.array(pd.get_dummies(self.img_labels_raw.iloc[,1])))
		self.img_dir = img_dir
		self.transform = Resize([300,300], antialias=True)

	def __len__(self):
		return len(self.img_labels)

	def __getitem__(self, idx):
		img_path = os.path.join(self.img_dir, self.img_labels_raw.iloc[idx, 0])
		image = read_image(img_path)
		label = self.img_labels[idx]
		image = self.transform(image)
		return image, label

training_data = ImagenetDataset(annotations_file = "/mnt/ceph_rbd/imagenet_data/imagenet_data_subset/train/image_labels.txt",
				img_dir = "/mnt/ceph_rbd/imagenet_data/imagenet_data_subset/train/")

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
print(f"Label: {label}")
