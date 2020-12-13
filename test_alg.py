from torchvision import models
from torchvision import transforms as T
import torch
from PIL import Image
import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from utils import visualize_game
from scipy.spatial.distance import cdist


data_dir = '/home/stanislaw/datasets/open-images/train'

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = T.Compose([
                        T.Resize(size=256),
                        T.CenterCrop(size=224),
                        T.ToTensor(),
                        T.Normalize(mean=mean, std=std)
                    ])

get_images = lambda im_id: transform(Image.open(os.path.join(data_dir, im_id)))

image_ids = [
        [
            'Bicycle/4eb478ec0835cfdd.jpg',
            'Bicycle/020feca9b536f1fe.jpg',
            'Balloon/8ff25f5ccc3b7717.jpg',
            'Cake/429a17e931a48e84.jpg',
            'Ball/978502eef1aa0d13.jpg',

            'Bicycle helmet/a7fa2eca5a25af76.jpg',
            'Computer monitor/1f5cf6e96f0e0690.jpg',
            'Bicycle helmet/0721a8637777a826.jpg',
            'Airplane/0a4abf0a8071b917.jpg',
            'Airplane/6fa1d2c9152bf37f.jpg'
        ],
        [
            'Airplane/0a4abf0a8071b917.jpg',
            'Bowl/0d1c49afcf00e948.jpg',
            'Airplane/6fa1d2c9152bf37f.jpg',
            'Alarm clock/46a0cd25a1f06d65.jpg',
            'Apple/69f3f3889793e68e.jpg',

            'Bowl/9f26942709846de3.jpg',
            'Bicycle helmet/a7fa2eca5a25af76.jpg',
            'Bowl/da95f58bf0813d29.jpg',
            'Bicycle helmet/0721a8637777a826.jpg',
            'Bicycle/4eb478ec0835cfdd.jpg',
        ],
        [
            'Human body/d50b0457d9624258.jpg',
            'Musical instrument/afe96a54fe0610cd.jpg',
            'Dog/c7a81616ee618ccd.jpg',
            'Rifle/24a5c73ad828580e.jpg',
            'Man/fa96d52b52a1ff28.jpg',

            'Man/200907829e4fbe95.jpg',
            'Musical keyboard/5e6b55b7203464b5.jpg',
            'Television/84a1048891bb636c.jpg',
            'Land vehicle/731f10ee0e1d6b70.jpg',
            'Vehicle/7c6e8d04b6e52053.jpg'
        ],
        [
            'Food/0bfcd180b1d67f94.jpg',
            'Man/fa96d52b52a1ff28.jpg',
            'Human body/d50b0457d9624258.jpg',
            'Television/84a1048891bb636c.jpg',
            'Vehicle/7c6e8d04b6e52053.jpg',

            'Dog/c7a81616ee618ccd.jpg',
            'Rifle/24a5c73ad828580e.jpg',
            'Man/200907829e4fbe95.jpg',
            'Giraffe/2647aeeff6ef00be.jpg',
            'Land vehicle/731f10ee0e1d6b70.jpg'
        ],
        [
            'Muffin/966ca10a7907662a.jpg',
            'Musical keyboard/0bff5d8cb2c2a2c9.jpg',
            'Musical instrument/39f032086de8375a.jpg',
            'Musical instrument/ffb2640fb0b5a865.jpg',
            'Musical instrument/ceda9adb4e88fc8e.jpg',

            'Musical instrument/5a31c61d001558c3.jpg',
            'Musical keyboard/0e160088b30f6234.jpg',
            'Musical instrument/7f467f785446806d.jpg',
            'Musical instrument/c929dbfa9cb5aa18.jpg',
            'Musical instrument/2eaca36ed49515ca.jpg'
        ],
        [
            'Human body/9eaffa603797e2dd.jpg',
            'Tree/7fa8b4a5f88946e4.jpg',
            'Bottle/2c9817c8076a3fd4.jpg',
            'Beer/e1b4b8f660967a1a.jpg',
            'Dog/04a27da7e880ddcd.jpg',
            
            'Dog/7cc816e323d254a0.jpg',
            'Giraffe/2647aeeff6ef00be.jpg',
            'Bottle/966d44c47effc043.jpg',
            'Bottle/7e2afdc21926bd14.jpg',
            'Dog/000c4d66ce89aa69.jpg'
        ],
        [
            'Unknown/1ee37d95b6c9a487.jpg',
            'Car/0b75807d9dc2f283.jpg',
            'Dog/7cc816e323d254a0.jpg',
            'Ice cream/75161733278e50d9.jpg',
            'Woman/3c678dab48dd6191.jpg',

            'Car/0c2db401e7e5bf89.jpg',
            'Car/0c9f9b713f229fba.jpg',
            'Human body/9eaffa603797e2dd.jpg',
            'Tree/7fa8b4a5f88946e4.jpg',
            'Dog/04a27da7e880ddcd.jpg'
            
        ],
        [
            'Unknown/5c4658dfb87ef8bf.jpg',
            'Car/0c9f9b713f229fba.jpg',
            'Human body/a013282071756b84.jpg',
            'Human body/3529ed57057b4d08.jpg',
            'Human body/e2eb9b5eca496b1e.jpg',
            
            
            'Human body/983255791f113009.jpg',
            'Human body/7772aaad3a1b72cc.jpg',
            'Human body/ada6444710b9b1c4.jpg',
            'Human body/3680bce5527b2efe.jpg',
            'Human body/4097eda463e6631c.jpg'
        ],
        [
            'Human body/9efc246660ca5253.jpg',
            'Television/84a1048891bb636c.jpg',
            'Land vehicle/731f10ee0e1d6b70.jpg',
            'Human body/3097b67f07b8f8e8.jpg',
            'Apple/69f3f3889793e68e.jpg',

            'Organ (Musical Instrument)/72ac183680821ed4.jpg',
            'Human body/1806f5b35279d7a5.jpg',
            'Human body/973c75e13ab55347.jpg',
            'Human body/969af219a0438f06.jpg',
            'Human body/434166afed88f797.jpg'
        ],
        [
            'Computer monitor/1f8d2f13a6cd1206.jpg',
            'Human body/9efc246660ca5253.jpg',
            'Television/84a1048891bb636c.jpg',
            'Land vehicle/731f10ee0e1d6b70.jpg',
            'Human body/3097b67f07b8f8e8.jpg',

            'Apple/69f3f3889793e68e.jpg',
            'Organ (Musical Instrument)/72ac183680821ed4.jpg',
            'Human body/1806f5b35279d7a5.jpg',
            'Human body/973c75e13ab55347.jpg',
            'Human body/969af219a0438f06.jpg'
        ],
        [
            'Building/0c5458ea3583233b.jpg',
            'Human body/a3ede9354b3f5cd5.jpg',
            'Human body/76571813b79399e5.jpg',
            'Human body/3097b67f07b8f8e8.jpg',
            'Ice cream/75161733278e50d9.jpg',
            
            'Human body/68b5e7af876a3062.jpg',
            'Human body/434166afed88f797.jpg',
            'Human body/226f45719bfa23b3.jpg',
            'Human body/1806f5b35279d7a5.jpg',
            'Unknown/1ee37d95b6c9a487.jpg' 
        ]
    ]

images = list(map(get_images, image_ids[3]))

# images = [transform(image) for image in images]
images = torch.stack(images)

model = models.resnet50(pretrained=True)
feature_extractor = torch.nn.Sequential(*(list(model.children())[:-2]))
feature_extractor.eval()  

f = feature_extractor(images).detach().cpu().numpy()

# print(f.reshape(f.shape[0], -1))

f = f.reshape(f.shape[0], -1)

pca = PCA(n_components=5)
# pca.fit(f[:5, :])
pca.fit(f)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
t_f = pca.transform(f)

print('Context\n', t_f[:5, :])
print('Source\n', t_f[5:, :])
std_ = np.std(t_f[5:, :] - t_f[5, :], axis=0)
std2_ = np.std(t_f[:5, :] - t_f[5, :], axis=0)
yhm = np.min(np.abs(t_f[:5, :] - t_f[5, :]), axis=0)
# print('EEEEEEEEEEE', t_f[:5, :] - t_f[5, :])
# std_ = np.mean(t_f[5:, :] - t_f[5, :], axis=0)
# std_ = np.sum(t_f[6:, :], axis=0)
# std_ = t_f[5, :] - std_
# std_ = np.mean(t_f[5:, :], axis=0)
# x = np.max(np.abs(std_))
x_i = np.argmax(np.abs(std_))
x = std_[x_i]
print('Std: ', std_)
print('close: ', yhm)
print('Std2: ', std2_)
print('x: ', x)
# if x < 0:
#     a_i = np.argmin(t_f[:5, x_i])
# else:
#     a_i = np.argmax(t_f[:5, x_i])
a_i = np.argmin(t_f[:5, x_i] - t_f[5, x_i])
a = t_f[a_i, x_i]
print('answer: ', a)
print('a_i: ', a_i)
print('x_i: ', x_i)

# idxs = list(reversed(np.argsort(std_)))[:2]
idxs = np.argsort(yhm)[:2]

per_row = f.shape[0] // 2
context = np.vstack(t_f[:per_row, :].reshape(per_row, -1))
source = np.vstack(t_f[per_row:, :].reshape(per_row, -1))
sims = cdist(context, source, 'euclidean')

print('SIMS\n', sims)
print(np.argmin(sims, axis=0))

plt.scatter(t_f[:5, idxs[0]], t_f[:5, idxs[1]], color='blue')
plt.scatter(t_f[5:, idxs[0]], t_f[5:, idxs[1]], color='red')

l = len(image_ids[0]) // 2
for i in range(l):
    plt.annotate(i, (t_f[i, idxs[0]], t_f[i, idxs[1]]))
    plt.annotate(i, (t_f[i+5, idxs[0]], t_f[i+5, idxs[1]]))

# visualize_game(images.permute(0, 2, 3, 1), 0, 0, save_f=None, show=True)

plt.show()

def pca_sims(f):
    f = f.reshape(f.shape[0], -1)
    pca = PCA(n_components=10)
    pca.fit(f)
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)
    t_f = pca.transform(f)
    per_row = f.shape[0] // 2
    context = np.vstack(t_f[:per_row].reshape(per_row, -1))
    source = np.vstack(t_f[per_row:].reshape(per_row, -1))
    sims = cdist(context, source, 'euclidean')

    return sims



