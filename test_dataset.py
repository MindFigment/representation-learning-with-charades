import torch
import torchvision
from torch.utils.data import Dataset
import csv
import os
from PIL import Image
import numpy as np


class TestDataset(Dataset):
    """
    Test dataset that includes image path when returning sample.
    Aguments:
        csv_file (string):
        root_dir (string):
        test_or_val (string): Whether it is test or validation dataset. 
                        It matters becaus in test we have only one choice, but in validation there can be more
    Returns:
        image: (PIL) if train
        (image, topics, choices): (PIL, list, list)
    """

    def __init__(self, csv_file='./csv_files/test.csv', root_dir='/media/STORAGE/DATASETS/open-images/challenge2018/', transform=None):
        self.images = self._process_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform


    def __len__(self):
        return len(self.images)

    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(os.path.join(self.root_dir, self.images[idx])).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        return image


    def _process_csv(self, csv_file):
        with open(csv_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            images = []
            for row in reader:
                images.append(row[0])
        return images


if __name__ == '__main__':
    
    transform = torchvision.transforms.ToTensor()
    data_dir  = '/media/STORAGE/DATASETS/open-images/challenge2018/'
    csv_file = './csv_files/test.csv'
    dataset = TestDataset(csv_file, data_dir, transform)
    dataloader = torch.utils.data.DataLoader(dataset)
    images = next(iter(dataloader))
    print(images.size())