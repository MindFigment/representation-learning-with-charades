import torch
import torchvision
from torch.utils.data import Dataset
import csv
import os
from PIL import Image
import numpy as np


class CharadesVal(Dataset):
    """
    Charades dataset that includes image path when returning sample.
    Aguments:
        csv_file (string):
        root_dir (string):
        test_or_val (string): Whether it is test or validation dataset. 
                        It matters becaus in test we have only one choice, but in validation there can be more
    Returns:
        image: (PIL) if train
        (image, topics, choices): (PIL, list, list)
    """

    def __init__(self, csv_file, root_dir, transform=None, test_or_val='test'):
        self.test_or_val = test_or_val
        self.data_dict = self._process_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        # print(len(self.data_dict['image_names']), len(self.data_dict['topics']))


    def __len__(self):
        return len(self.data_dict['topics'])

    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        get_image = lambda img_name: Image.open(os.path.join(self.root_dir, img_name)).convert('RGB')

        images = list(map(get_image, self.data_dict['image_names'][idx]))
        
        if self.transform:
            images = list(map(lambda image: self.transform(image), images))

        images = torch.stack(images)

        return images, self.data_dict['topics'][idx], self.data_dict['choices'][idx]


    def _process_csv(self, csv_file):
        if self.test_or_val == 'val':
            process_choices = lambda row: [int(choice) for choice in row['PlayerChoice'].split(',')]
            process_topics = lambda row: [int(topic) for topic in row['Topic'].split(',')]
        elif self.test_or_val == 'test':
            process_choices = lambda row: int(row['PlayerChoice'])
            process_topics = lambda row: int(row['Topic'])
        else:
            raise ValueError('Test_or_val can be only val or test, but got {}!'.format(self.test_or_val))

        all_images = []
        all_topics = []
        all_choices = []
        with open(csv_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                images = list(row.values())[:10]
                choices = process_choices(row)
                topics = process_topics(row)

                all_images.append(images)
                all_topics.append(topics)
                all_choices.append(choices)

        return {'image_names': all_images, 'topics': all_topics, 'choices': all_choices}


if __name__ == '__main__':
    
    transform = torchvision.transforms.ToTensor()
    data_dir  = '/home/stanislaw/datasets/open-images/train/'
    csv_file = './validation_labeled.csv'
    dataset = CharadesVal(csv_file, data_dir, transform)
    dataloader = torch.utils.data.DataLoader(dataset)
    images, topics, choices = next(iter(dataloader))
    print(topics, choices)