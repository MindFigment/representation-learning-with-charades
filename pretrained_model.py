from torchvision import models
from torchvision import transforms as T
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torchvision
from torch import nn
import torch.nn.functional as F


def load_pretrained(model_name, drop_last):
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True) 
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif model_name == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif model_name == 'inception':
        model = models.inception(pretrained=True)
    else:
        raise ValueError('Wrong model name: {}'.format(model_name))

    # feature_extractor = model._modules.get(layer_name)
    feature_extractor = torch.nn.Sequential(*(list(model.children())[:-drop_last]))

    feature_extractor.eval()
    
    return feature_extractor


def get_pretrained_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    image_transform = T.Compose([T.Resize(size=256),
                                 T.CenterCrop(size=224),
                                 T.ToTensor(),
                                 T.Normalize(mean=mean, std=std)
    ])

    return image_transform






def main():
    model = load_pretrained('alexnet', drop_last=2)
    # model = models.vgg16(pretrained=True)
    # model = models.alexnet(pretrained=True)
    # model = models.inception_v3(pretrained=True)
    # model = models.resnet50(pretrained=True)
    #  modules = model._modules.get('layer3')._modules.get('1')._modules.get('conv1')
    # modules = model._modules
    # print(list(model.children()))
    for child in model.children():
        print('Child: {}'.format(child))

    # newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))



if __name__ == '__main__':
    main()

