from pretrained_model import load_pretrained, get_pretrained_transform
from utils import visualize_similarities, norm, title_from_dict
from charades_dataset import CharadesVal
from algorithms import compute_similarities, choose_answer, choose_answer2, choose_answer3, choose_dict, choose_dict2, choose_dict3

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import csv
import numpy as np
import click
from tqdm import tqdm
import os

@torch.no_grad()
# @click.command()
# @click.option('--per_row', '-pr', default=5, help='How many context images in every example')
# @click.option('--metric', '-me', default='cosine', help='How to measure similarity between vectors')
# @click.option('--model_name', '-mn', default='alexnet', help='Which model to use')
# @click.option('--drop_last', '-dl', default=2, help='If pertrained is true, specify how many last layers to drop')
# @click.option('--root_dir', '-rd', default='/home/stanislaw/datasets/open-images/train', help='Dir with visualizations with repaired labels')
# @click.option('--csv_test', '-csv_t', default='./csv_files/handpicked.csv', help='Dir with visualizations with repaired labels')
# @click.option('--csv_val', '-csv_v', default='./csv_files/validation_labeled.csv', help='Dir with visualizations with repaired labels')
# @click.option('--batch_test', '-bt', default=1)
# @click.option('--batch_val', '-bv', default=1)
# @click.option('--vis_val/--no-show_val', default=False, help='Do you want to plot each val example')
# @click.option('--vis_test/--no-show_test', default=True, help='Do you want to plot each test example')
# @click.option('--save_val', default='./vis_val/check-cosine', help='Do you want to visualize val data')
# @click.option('--save_test', default='./vis_test/check-cosine', help='Do you want to visualize test data')
def validate(per_row=5,
             metric='cosine',
             model='alexnet',
             drop_last=2,
             root_dir='/home/stanislaw/datasets/open-images/train',
             csv_test='./csv_files/handpicked.csv',
             csv_val='./csv_files/validation_labeled.csv',
             batch_test=1,
             batch_val=1,
             vis_val=False, 
             vis_test=True,
             save_val='./vis_val/check-cosine',
             save_test='./vis_test/check-cosine',
             algorithm='choose_dict'):

    transform = get_pretrained_transform()

    test_dataset_params = {
        'csv_file': csv_test,
        'root_dir': root_dir,
        'transform': transform,
        'test_or_val': 'test'
    }

    val_dataset_params = {
        'csv_file': csv_val,
        'root_dir': root_dir,
        'transform': transform,
        'test_or_val': 'val'
    }

    test_dataloader_params = {
        'batch_size': batch_test,
        'shuffle': False,
        'num_workers': 1
    }

    val_dataloader_params = {
        'batch_size': batch_val,
        'shuffle': False,
        'num_workers': 1
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_dataset = CharadesVal(**test_dataset_params)
    val_dataset = CharadesVal(**val_dataset_params)

    test_dataloader = DataLoader(test_dataset, **test_dataloader_params)
    val_dataloader = DataLoader(val_dataset, **val_dataloader_params)

    if type(model) is str and model in ['alexnet', 'resnet50', 'inception', 'vgg16']:
        model = load_pretrained(model, drop_last=drop_last)
    # else:
    #     raise ValueError('For now we do not have our own model')

    if algorithm not in ['choose_dict', 'choose_dict2', 'choose_dict3']:
        raise ValueError('Wrong algorithm {}!'.format(algorithm))
    else:
        if algorithm == 'choose_dict':
            get_answer_dict = choose_dict
        elif algorithm == 'choose_dict2':
            get_answer_dict = choose_dict2
        else:
            get_answer_dict = choose_dict3

    try:
        vis_val_dir = os.path.join(*vis_val.split('/')[:-1])
        os.makedirs(vis_val_dir)
    except FileExistsError:
        print('{} already exists!'.format(vis_val_dir))

    try:
        vis_test_dir = os.path.join(*vis_test.split('/')[:-1])
        os.makedirs(vis_test_dir)
    except FileExistsError:
        print('{} already exists!'.format(vis_test_dir))

    
    ######################
    ##### VALIDATION #####
    ######################

    # Count number of examples to compute accuracy
    n_val = 0
    correct_val = 0
    i = 1
    for batch, topics, choices in tqdm(val_dataloader):
        # Batch is always of size: (1, 10, 3, 224, 224) so we take batch[0] to get size (10, 3, 224, 224)
        # batch = batch.view(-1, *batch.size()[-3:])
        for example in batch:
            example = example.to(device)
            features = model(example).cpu()
            sims = compute_similarities(features, metric=metric)

            answer_dict = get_answer_dict(sims)

            # answers = [choose_answer3(sims, choice) for choice in choices]
            answers = [answer_dict[int(choice)] for choice in choices]
            n_val += len(answers)
            correct_val += np.sum(np.array(topics) == np.array(answers))

            if save_val:
                # Normalize images before plotting 
                batch = norm(example)
                title = title_from_dict(answer_dict)
                visualize_similarities(example.cpu(), sims, answer_dict, metric=metric, title=title, show=False, save_f=vis_val + str(i))
            i += 1

    acc_val = correct_val / n_val

    ######################
    ######## TEST ########
    ######################

    n_test = 0
    correct_test = 0
    i = 1
    for batch, topic, choice in tqdm(test_dataloader):
        # Batch is always of size: (1, 10, 3, 224, 224) so we take batch[0] to get size (10, 3, 224, 224)
        batch = batch.view(-1, *batch.size()[-3:])
        batch = batch.to(device)
        features = model(batch).cpu()
        sims = compute_similarities(features, metric=metric)

        # answer = choose_answer(sims, choice)
        answer_dict = get_answer_dict(sims)
        answer = answer_dict[int(choice)]
        n_test += 1
        correct_test += int(answer == topic)

        if save_test:
            # Normalize images before plotting 
            batch = norm(batch)
            title = title_from_dict(answer_dict)
            visualize_similarities(batch.cpu(), sims, answer_dict, metric=metric, title=title, show=False, save_f=vis_test + str(i))
        i += 1

    acc_test = correct_test / n_test
    
    return acc_val, acc_test


if __name__ == '__main__':

    params = {
        'per_row': 5,
        'metric': 'cosine',
        'model': 'resnet50',
        'drop_last': 2,
        'root_dir': '/home/stanislaw/datasets/open-images/train',
        'csv_test': './csv_files/handpicked.csv',
        'csv_val': './csv_files/validation_labeled.csv',
        'batch_test': 1,
        'batch_val': 1,
        'save_test': True,
        'save_val': True,
        'vis_val': './vis_val/check-cosine-2-last2-',
        'vis_test': './vis_test/check-cosine-2-last2-',
        'algorithm': 'choose_dict2'
    }

    acc_val, acc_test = validate(**params)

    print('Val acc: {}'.format(acc_val))
    print('Test acc: {}'.format(acc_test))