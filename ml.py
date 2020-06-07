from torchvision import models
from scipy.spatial.distance import cdist
import numpy as np
import torch
import torchvision.transforms as T
from vae import VAE


def get_model(model_name, drop_last=2):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if model_name == 'vae':
        model = VAE(latent_dim=200, nf=128)#.to(device)
        feature_extractor = model.get_feature_extractor()
    else:
        model = models.resnet50(pretrained=True)
        feature_extractor = torch.nn.Sequential(*(list(model.children())[:-drop_last]))
        feature_extractor.eval()
    return feature_extractor


def compute_similarities(features, metric='cosine'):
    per_row = features.shape[0] // 2
    context = np.vstack(features[:per_row].reshape(per_row, -1))
    source = np.vstack(features[per_row:].reshape(per_row, -1))
    sims = cdist(context, source, metric)
    return sims


def get_transform(model_name='vae'):
    if model_name == 'vae':
        image_transform = T.Compose([T.Resize(size=256),
                                     T.CenterCrop(size=224),
                                     T.ToTensor()
    ])
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image_transform = T.Compose([T.Resize(size=256),
                                    T.CenterCrop(size=224),
                                    T.ToTensor(),
                                    T.Normalize(mean=mean, std=std)
        ])
    return image_transform


def choose_dict(sims):
    min_ = np.argmin(sims, axis=0)
    choices_dict = dict(zip(list(range(len(min_))), min_))
    return choices_dict


def choose_dict2(sims):
    per_row = sims.shape[0]
    # Sort distances descending, keep index
    x = np.argsort(sims.reshape(-1))
    source = x % per_row
    context = x // per_row
    computerChoice = []
    playerChoice = []
    choices_dict = {}
    for i in range(len(source)):
        if len(playerChoice) >= per_row:
            break
        if source[i] not in playerChoice and context[i] not in computerChoice:
            choices_dict[source[i]] = context[i]
            playerChoice.append(source[i])
            computerChoice.append(context[i])
    # print('Choices: {}'.format(choices))
    return choices_dict


def choose_dict3(sims, temperature=0.1):
    per_row = sims.shape[0]
    # Sort distances descending, keep index
    x = np.argsort(sims.reshape(-1))
    # Sort distances descending, keep distance
    sorted_ = np.sort(sims.reshape(-1))
    source = x % per_row
    context = x // per_row
    computerChoice = []
    playerChoice = []
    choices_dict = {}
    distances_dict = {}
    for i in range(len(source)):
        if len(playerChoice) >= per_row:
            break
        if source[i] not in playerChoice and context[i] in computerChoice and (sorted_[i] - distances_dict[context[i]]) < temperature:
            choices_dict[source[i]] = context[i]
            playerChoice.append(source[i])
        elif source[i] not in playerChoice and context[i] not in computerChoice:
            choices_dict[source[i]] = context[i]
            playerChoice.append(source[i])
            computerChoice.append(context[i])
            distances_dict[context[i]] = sorted_[i]
    # print('Choices: {}'.format(choises))
    return choices_dict


@torch.no_grad()
def inference(images, model, choice, metric, alg, transform):
    # transform = get_transform('pretrained')
    images = [transform(images) for images in images]
    images = torch.stack(images)
    features = model(images)
    sims = compute_similarities(features, metric=metric)
    answer_dict = alg(sims)
    return answer_dict[choice]