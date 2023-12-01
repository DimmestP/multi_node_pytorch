import os
import torch
import pandas
from torchvision.io import read_image

annotations_file = "/mnt/ceph_rbd/imagenet_data/imagenet_data_subset/train/image_labels.txt"
img_dir = "/mnt/ceph_rbd/imagenet_data/imagenet_data_subset/train/"

img_labels = pd.read_csv(annotations_file, sep='\t')

for i in range(1,len(img_labels)):
	img_path = os.path.join(img_dir, img_labels.iloc[i, 0])
	image = read_image(img_path)
	if image.size()[0] != 3:
		os.remove(img_path)
