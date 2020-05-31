import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import math
from collections import Counter, defaultdict
from torchvision.utils import make_grid



def norm(tensors):
    for t in tensors:
        minimum = t.min()
        maximum = t.max()
        t.add_(-minimum).div_(maximum - minimum + 1e-5)

    return tensors


def title_from_dict(d):
    title = ' | '.join(map(lambda x: str(x[0]) + ':' + str(x[1]), d.items()))
    return title


def visualize_similarities(images, sims, choose_dict, metric, title, show=False, save_f=None):
    per_row = sims.shape[0]
    images = images.permute(0, 2, 3, 1).numpy()
    _, axes = plt.subplots(per_row + 1, per_row, figsize=(per_row * 5, (per_row + 1) * 5))
    for i in range(per_row):
        axes[0, i].imshow(images[i + per_row])
        axes[0, i].axis('off')
        for j in range(per_row):
            axes[j + 1, i].imshow(images[j])
            axes[j + 1, i].axis('off')
            if choose_dict[i] == j:
                axes[j + 1, i].set_title('{0:.3f}'.format(sims[j, i]), backgroundcolor='lightcoral', size=30)
            else:
                axes[j + 1, i].set_title('{0:.3f}'.format(sims[j, i]), size=30)

    plt.suptitle('{} similarty \n{}'.format(metric, title), size=60)
    
    if save_f:
        # print('Saving {}'.format(save_f))
        plt.savefig(save_f)
        # print('Done!')

    if show:
        plt.subplots_adjust(left=0.36, bottom=0.11, right=1-0.36, top=1-0.11, wspace=0.03, hspace=0.2)
        plt.show()
    else:
        plt.close('all')


def visualize_game(images, index, topic, save_f=None, show=False):
    if images is np.ndarray:
        per_row = images.shape[0] // 2
    else:
        per_row = len(images) // 2
    _, axes = plt.subplots(2, per_row, figsize=(5 * per_row, 7))
    mid = per_row // 2
    for i in range(per_row):
        axes[0, i].imshow(images[i])
        axes[1, i].imshow(images[i + per_row])
        if i == mid:
            axes[0, i].set_title('Context')
            axes[1, i].set_title('Source')
        if i == topic:
            axes[0, i].spines['bottom'].set_color('red')
            axes[0, i].spines['top'].set_color('red')
            axes[0, i].spines['right'].set_color('red')
            axes[0, i].spines['left'].set_color('red')

            axes[0, i].spines['bottom'].set_linewidth(4)
            axes[0, i].spines['top'].set_linewidth(4)
            axes[0, i].spines['right'].set_linewidth(4)
            axes[0, i].spines['left'].set_linewidth(4)

            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])

            axes[1, i].axis('off')
        else:
            axes[0, i].axis('off')
            axes[1, i].axis('off')

        plt.suptitle('Example index: {}'.format(index))
        
            
    if save_f:
        plt.savefig(save_f)

    if show:
        plt.show()
    else:
        plt.close('all')


def visualize_labels(images, index, topics, choices, save_f=None, show=False):
    if images is np.ndarray:
        per_row = images.shape[0] // 2
    else:
        per_row = len(images) // 2
    _, axes = plt.subplots(2, per_row, figsize=(5 * per_row, 7))
    mid = per_row // 2

    colors = ['red', 'purple', 'blue', 'green', 'orange']
    line_width = 5

    c_counter = Counter(topics)
    s_counter = Counter(choices)
    c_dict = defaultdict(list)
    s_dict = defaultdict(list)
    for i in range(len(topics)):
        c_dict[topics[i]].append(colors[i])
        s_dict[choices[i]].append(colors[i])

    for i in range(per_row):
        axes[0, i].imshow(images[i])
        axes[1, i].imshow(images[i + per_row])
        if i == mid:
            axes[0, i].set_title('Context')
            axes[1, i].set_title('Source')
        if i in topics:
            cols = c_dict[i]
            if c_counter[i] == 1:
                axes[0, i].spines['bottom'].set_color(cols[0])
                axes[0, i].spines['top'].set_color(cols[0])
                axes[0, i].spines['right'].set_color(cols[0])
                axes[0, i].spines['left'].set_color(cols[0])
            elif c_counter[i] == 2:
                axes[0, i].spines['bottom'].set_color(cols[0])
                axes[0, i].spines['top'].set_color(cols[0])
                axes[0, i].spines['right'].set_color(cols[1])
                axes[0, i].spines['left'].set_color(cols[1])
            elif c_counter[i] == 3:
                axes[0, i].spines['bottom'].set_color(cols[0])
                axes[0, i].spines['top'].set_color(cols[0])
                axes[0, i].spines['right'].set_color(cols[1])
                axes[0, i].spines['left'].set_color(cols[2])
            elif c_counter[i] == 4:
                axes[0, i].spines['bottom'].set_color(cols[0])
                axes[0, i].spines['top'].set_color(cols[3])
                axes[0, i].spines['right'].set_color(cols[0])
                axes[0, i].spines['left'].set_color(cols[2])
            else:
                raise ValueError('Too many labels for context {}!'.format(i))

            axes[0, i].spines['bottom'].set_linewidth(line_width)
            axes[0, i].spines['top'].set_linewidth(line_width)
            axes[0, i].spines['right'].set_linewidth(line_width)
            axes[0, i].spines['left'].set_linewidth(line_width)

            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])
        else:
            axes[0, i].axis('off')

        if i in choices:
            cols = s_dict[i]
            if s_counter[i] == 1:
                axes[1, i].spines['bottom'].set_color(cols[0])
                axes[1, i].spines['top'].set_color(cols[0])
                axes[1, i].spines['right'].set_color(cols[0])
                axes[1, i].spines['left'].set_color(cols[0])
            elif s_counter[i] == 2:
                axes[1, i].spines['bottom'].set_color(cols[0])
                axes[1, i].spines['top'].set_color(cols[0])
                axes[1, i].spines['right'].set_color(cols[1])
                axes[1, i].spines['left'].set_color(cols[1])
            elif s_counter[i] == 3:
                axes[1, i].spines['bottom'].set_color(cols[0])
                axes[1, i].spines['top'].set_color(cols[0])
                axes[1, i].spines['right'].set_color(cols[1])
                axes[1, i].spines['left'].set_color(cols[2])
            elif s_counter[i] == 4:
                axes[1, i].spines['bottom'].set_color(cols[0])
                axes[1, i].spines['top'].set_color(cols[3])
                axes[1, i].spines['right'].set_color(cols[1])
                axes[1, i].spines['left'].set_color(cols[2])
            else:
                raise ValueError('Too many labels for source {}!'.format(i))

            axes[1, i].spines['bottom'].set_linewidth(line_width)
            axes[1, i].spines['top'].set_linewidth(line_width)
            axes[1, i].spines['right'].set_linewidth(line_width)
            axes[1, i].spines['left'].set_linewidth(line_width)

            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])
        else:
            axes[1, i].axis('off')

        plt.suptitle('Example index: {}'.format(index))
        
            
    if save_f:
        plt.savefig(save_f)

    if show:
        plt.show()
    else:
        plt.close('all')
