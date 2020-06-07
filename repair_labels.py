import csv
import click
import os
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

from utils import visualize_labels

@click.command()
@click.option('--root', '-r', default='/media/STORAGE/DATASETS/open-images/', help='Dir where your data sits')
@click.option('--csv_file', '-csv', default='./csv_files/validation_labeled.csv', help='Csv file with correct labels')
@click.option('--output_dir', '-od', default='/media/STORAGE/DATASETS/open-images/validation', help='Dir with visualizations with repaired labels')
def main(root, csv_file, output_dir):
    data_dir = os.path.join(root, 'train')

    transform = T.Compose([
                        T.Resize(size=224),
                        T.CenterCrop(size=224)
    ])

    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in tqdm(enumerate(reader)):
            choices = [int(choice) for choice in row['PlayerChoice'].split(',')]
            topics = [int(topic) for topic in row['Topic'].split(',')]
            image_ids = list(row.values())[:10]
            # print(part_paths)
            # print('Topic: {}, Player choice: {}, ({})'.format(topics, choices, len(topics) == len(choices)))
            get_images = lambda im_id: transform(Image.open(os.path.join(data_dir, im_id)))
            images = list(map(get_images, image_ids))
            save_f = os.path.join(output_dir, 'example' + str(i + 1))
            visualize_labels(images, i + 1, topics, choices, save_f, show=False)


if __name__ == '__main__':
    main()