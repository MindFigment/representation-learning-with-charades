from torchvision import models
from scipy.spatial.distance import cdist
import numpy as np
import torch
import torchvision.transforms as T



def get_model(drop_last=2):
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


def get_pretrained_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_transform = T.Compose([T.Resize(size=256),
                                 T.CenterCrop(size=224),
                                 T.ToTensor(),
                                 T.Normalize(mean=mean, std=std)
    ])
    return image_transform


def choose_answer(sims, choice):
    min_ = np.argmin(sims[:, choice])
    return min_


@torch.no_grad()
def inference(images, model, choice, metric):
    transform = get_pretrained_transform()
    images = [transform(images) for images in images]
    images = torch.stack(images)
    features = model(images)
    sims = compute_similarities(features, metric=metric)
    answer = choose_answer(sims, choice)
    return answer