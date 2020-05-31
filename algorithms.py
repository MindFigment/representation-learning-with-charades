from scipy.spatial.distance import cdist
import numpy as np


######################
##### SIMILARITY #####
######################

def compute_similarities(features, metric='cosine'):
    per_row = features.shape[0] // 2
    context = np.vstack(features[:per_row].reshape(per_row, -1))
    source = np.vstack(features[per_row:].reshape(per_row, -1))
    sims = cdist(context, source, metric)

    return sims


###############################################
##### CHOOSE ANSWER BASED ON SIMILARITIES #####
###############################################

def choose_answer(sims, choice):
    min_ = np.argmin(sims[:, choice])
    return min_


def choose_answer2(sims, choice):
    x = np.argsort(sims.reshape(-1))
    a = x % 5
    b = x // 5
    used = []
    chosen = []
    choises = {}
    for i in range(len(a)):
        if a[i] not in chosen and b[i] not in used:
            choises[a[i]] = b[i]
            chosen.append(a[i])
            used.append(b[i])

    # print('Choices: {}'.format(choises))
    return choises[int(choice)]


def choose_answer3(sims, choice):
    x = np.argsort(sims.reshape(-1))
    a = x % 5
    b = x // 5
    temperature = 0.1
    sorted_ = np.sort(sims.reshape(-1))
    used = []
    chosen = []
    choises = {}
    distances = {}
    for i in range(len(a)):
        if a[i] not in chosen and b[i] in used and x[i] - distances[b[i]] < temperature:
            choises[a[i]] = b[i]
            chosen.append(a[i])
        elif a[i] not in chosen and b[i] not in used:
            choises[a[i]] = b[i]
            chosen.append(a[i])
            used.append(b[i])
            distances[b[i]] = sorted_[x[i]]

    # print('Choices: {}'.format(choises))
    return choises[int(choice)]


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


def choose_dict3(sims):
    per_row = sims.shape[0]
    # Sort distances descending, keep index
    x = np.agsort(sims.reshape(-1))
    # Sort distances descending, keep distance
    sorted_ = np.sort(sims.reshape(-1))
    source = x % per_row
    context = x // per_row
    temperature = 0.1
    computerChoice = []
    playerChoice = []
    choices_dict = {}
    distances_dict = {}
    for i in range(len(source)):
        if len(playerChoice) >= per_row:
            break
        if source[i] not in playerChoice and context[i] in computerChoice and x[i] - distances_dict[context[i]] < temperature:
            choices_dict[source[i]] = context[i]
            playerChoice.append(source[i])
        elif source[i] not in playerChoice and context[i] not in computerChoice:
            choices_dict[source[i]] = context[i]
            playerChoice.append(source[i])
            computerChoice.append(context[i])
            distances_dict[context[i]] = sorted_[x[i]]

    # print('Choices: {}'.format(choises))
    return choices_dict