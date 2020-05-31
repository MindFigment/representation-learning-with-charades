import torch
import torchvision
from image_folder_with_paths import ImageFolderWithPaths
import torchvision.transforms as T
import csv
import pandas as pd
from utils import visualize_game
import click
import numpy as np
import os
from tqdm import tqdm


@click.command()
@click.option('--mode', '-m', type=click.Choice(['a', 'w'], case_sensitive=False), default='a', help='Create new or append to existing validation file')
@click.option('--num', '-n', default=10, help='Number of examples to generate')
@click.option('--output', '-o', default='./validation.csv', help='File with labeled examples')
@click.option('--seed', '-s', default=0, help='Set seed if you want to get reproducible examples, by default set to 0. If you want random examples every time set seed to -1')
@click.option('--root', '-r', default='/home/stanislaw/datasets/open-images', help='Dir where your data sits')
def main(mode, num, output, seed, root):
    if seed != -1:
        torch.manual_seed(seed)
        np.random.seed(seed)

    transform = T.Compose([
                           T.Resize(size=224),
                           T.CenterCrop(size=224),
                           T.ToTensor(),
    ])

    data_dir = os.path.join(root, 'train')
    dataset = ImageFolderWithPaths(data_dir, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
    iter_dataloader = iter(dataloader)

    # Column names for validation.csv file
    context = [''.join(['Context', str(i)]) for i in range(5)]
    source = [''.join(['Source', str(i)]) for i in range(5)]
    columns = [*context, *source, 'PlayerChoice', 'Topic']

    # Variable to keep track from which row append, if we write
    # to file, not append, then we start from 0. We need to info
    # to properly add index to our example file name (for example: example10.jpg)
    # We start from index 1 because we want to enumarete examples starting from 1
    # and when counting rows we take into accout columns row (for example: if 
    # there is only one example then start will be equal to 2)
    start = 1
    # If we want to append to exisiting file then we need to know how
    # many lines are already in our file
    if mode == 'a' and os.path.isfile(output):
        with open(output, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            # Count how many rows to append properly next examples
            for _ in reader:
                start += 1
    with open(output, mode) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        # If start == 1 it means we have empty file
        if mode == 'w' or start == 1:
            # Add headers to the file if we create new or override existing file
            writer.writeheader()

        # If start is greater then 1 it means we are appending to existing one
        # Also if seed is different from -1 it means we want to generate reproducible
        # sequence of examples so to properly append we need to omit start-1 number of batches
        if start > 1 and seed != -1:
            for _ in range(start - 1):
                _ = next(iter_dataloader)

        topics = np.random.choice(5, start - 1 + num)

        for i in tqdm(range(start - 1, start - 1 + num)):
            images, _, paths = next(iter_dataloader)

            # Leave only last dir on the path to the image (for example: Human/image.jpg)
            part_paths = list(map(lambda path: '/'.join(path.split('/')[-2:]), paths))

            # Randomly choose the topic for this example
            topic = topics[start - 1 + i]

            # Values for every column, last is set to 'None', because we will
            # fill the information about correct answers manually
            row = [*part_paths, 'None', topic]
            
            row_dict = dict(zip(columns, row))
            writer.writerow(row_dict)
            
            save_f = os.path.join(root, 'validation', 'example' + str(i + 1))

            # Create image of our particular examples to help us visually inspect it
            # and decide upon correct label we want to assign to it
            visualize_game(images.permute(0, 2, 3, 1).numpy(), index=i + 1, topic=topic, show=False, save_f=save_f)

    # with open('output.csv', 'r') as csvfile:
    #     reader = csv.DictReader(csvfile)
    #     for row in reader:
    #         weights = [(int(score.split(' ')[0]), float(score.split(' ')[1])) for score in row['Weights'].split(',')]
    #         print(row['Topic'], row['Weights'])
    #         print('Topic: {}, Weights: {}'.format(row['Topic'], weights))


if __name__ == '__main__':
    main()
