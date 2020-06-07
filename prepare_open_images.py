import os
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


def compute_largest_area(image_id, label_data):
    largest = 0
    class_id = None
    for d in label_data:
        area = (d['XMax'] - d['XMin']) * (d['YMax'] - d['YMin'])
        if area > largest:
            largest = area
            class_id = d['LabelName']
    return class_id, largest


def main():
    cvs_path = '/media/STORAGE/DATASETS/open-images/'
    class_ids_path = os.path.join(cvs_path, 'class-descriptions-boxable.csv')
    images_info_path = os.path.join(cvs_path, 'validation-annotations-bbox.csv')

    class_ids = pd.read_csv(class_ids_path, header=None)
    images_info = pd.read_csv(images_info_path)

    cl_keys = class_ids.iloc[:, 0]
    cl_values = class_ids.iloc[:, 1]
    cl_dict = dict(zip(cl_keys, cl_values))

    im_keys = list(images_info['ImageID'])
    print('Unique image ids: {}'.format(len(set(im_keys))))
    im_values = images_info[['LabelName', 'XMin', 'XMax', 'YMin', 'YMax']].to_dict('records')
    im_dict = defaultdict(list)
    for k, v in zip(im_keys, im_values):
        im_dict[k].append(v)

    label_counts = defaultdict(int)
    img2label = dict()
    for image_id, label_data in im_dict.items():
        class_id, largest = compute_largest_area(image_id, label_data)
        # print('Label: {}, Area: {}'.format(cl_dict[label], largest))
        label = cl_dict[class_id]
        img2label[image_id] = label
        label_counts[label] += 1

    for label in tqdm(label_counts.keys()):
        dir_path = os.path.join(cvs_path, 'train', label)
        try:
            os.makedirs(dir_path)
            # print('Created dir: {}'.format(dir_path))
        except FileExistsError:
            print('{} already exists!'.format(dir_path))

    dir_from = os.path.join(cvs_path, 'validation')
    for image_id, label in tqdm(img2label.items()):
        im = image_id + '.jpg'
        dir_to = os.path.join(cvs_path, 'train', label, im)
        dir_from_full = os.path.join(dir_from, im)
        try:
            os.replace(dir_from_full, dir_to)
        except FileNotFoundError:
            print('No such file: {}'.format(image_id))

    # Create folder unknown for all images with no labels that are left
    # in oryginal validation folder
    unknown_dir = os.path.join(cvs_path, 'train', 'Unknown')
    try:
        os.makedirs(unknown_dir)
    except FileExistsError:
        print('{} already exists!'.format(dir_path))

    left_images = os.listdir(dir_from)

    for im in left_images:
        dir_to = os.path.join(unknown_dir, im)
        dir_from_full = os.path.join(dir_from, im)
        try:
            os.replace(dir_from_full, dir_to)
        except FileNotFoundError:
            print('No such file: {}'.format(im))


if __name__ == '__main__':
    main()
