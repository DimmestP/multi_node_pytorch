import os
import torch
import pandas as pd
from torchvision.io import read_image

# Remove grey scale training images

annotations_file = "/mnt/ceph_rbd/imagenet_data/imagenet_data_subset/train/image_labels.txt"
img_dir = "/mnt/ceph_rbd/imagenet_data/imagenet_data_subset/train/"

img_labels = pd.read_csv(annotations_file, sep='\t')

for i in range(1,len(img_labels)):
	img_path = os.path.join(img_dir, img_labels.iloc[i, 0])
	image = read_image(img_path)
	if image.size()[0] != 3:
		os.remove(img_path)

# Remove grey scale validation images

annotations_file = "/mnt/ceph_rbd/imagenet_data/imagenet_data_subset/validation/image_labels.txt"
img_dir = "/mnt/ceph_rbd/imagenet_data/imagenet_data_subset/validation/"

img_labels = pd.read_csv(annotations_file, sep='\t')

greyscale_images = []

for i in range(1,len(img_labels)):
        img_path = os.path.join(img_dir, img_labels.iloc[i, 0])
        image = read_image(img_path)
        if image.size()[0] != 3:
                greyscale_images.append(i)

print(image_labels)
