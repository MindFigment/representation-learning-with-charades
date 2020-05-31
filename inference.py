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


@torch.no_grad()
def inference(images, model, choice, metric, save_f=None):
    features = model(images)

    print('FEATURES')
    f = np.sqrt(np.sum(np.power(features.reshape(features.shape[0], -1).numpy(), 2), axis=1))
    print(f)

    sims = compute_similarities(features, metric=metric)

    print('SIMS')
    print(sims)

    choose_d = choose_dict(sims)
    answer = choose_d[choice]

    if save_f:
        images = norm(images)
        visualize_similarities(images, sims, choose_d, title='Choice: {} | Answer: {}'.format(choice, answer), metric=metric, show=False, save_f=save_f)

    return answer


if __name__ == '__main__':

    batch_size = 8

    transform = get_pretrained_transform()

    dataset_params = {
        'root': '~/datasets/open-images/train',
        'transform': transform
    }

    dataloader_params = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': 1
    }

    dataset = ImageFolder(**dataset_params)

    iter_data_loader = iter(DataLoader(dataset, **dataloader_params))

    images = next(iter_data_loader)[0]

    model = load_pretrained("resnet50", 2)

    choice = np.random.choice(batch_size // 2)

    r = np.random.choice(1000)
    metric = 'cosine' #'chebyshev' # 'canberra' #'braycurtis' #'dice' #'correlation'
    save_f = './random_inference_visualizations/random_' + metric + str(r)
    # save_f=None

    answer = inference(images, model, choice, metric=metric, save_f=save_f)

    print('Answer: {}'.format(answer))