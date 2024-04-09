import numpy as np
import parse
from itertools import combinations
from register import HandRegister
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings

from experiment_values import PARTICIPANT_ID

warnings.filterwarnings("ignore")

# load data
input_file = './calibration-results/{}.txt'.format(PARTICIPANT_ID)
with open(input_file, 'r') as f:
    lmk, lmk_world, trigger = parse.parse(f.read())

# flatten 21 x 3 features into a vector of length 63
lmk = lmk.reshape((-1, trigger.size))

# cut stream of data into trials
idx_trial_start, y, trial_length = parse.split_trial(trigger=trigger)
trial_length = min(trial_length)
x_raw = np.stack([lmk[:, idx: idx + trial_length] for idx in idx_trial_start], axis=-1)

# dimensions: features (21 x 3), trial
x_raw = x_raw.mean(axis=1)
x_raw.shape

hand_reg = HandRegister()
hand_reg.fit(x_raw)
x = hand_reg.transform(x_raw)

# todo: this should be standard above
x_sklearn = x.T

# Group by trial 
label_vals = np.unique(y)

# Initialize dictionary to store separated data
separated_data = {label: [] for label in label_vals}

# Iterate over trials and separate data based on labels
for trial, label in zip(x.T, y):
    separated_data[label].append(trial)

# Convert lists to arrays
for label, trials in separated_data.items():
    separated_data[label] = np.array(trials).reshape(7, 3 * 21)

n_splits = 3

k_fold = StratifiedKFold(n_splits=n_splits)

norm_class = KNeighborsClassifier(n_neighbors=1)

def find_best_combo2(remaining_labels):
    acc_dict = dict()
    for label_to_rm in remaining_labels:
        # get a bool_idx of all active ones
        bool_idx = np.array([_y != label_to_rm for _y in y])
        _x = x_sklearn[bool_idx, :]
        _y = y[bool_idx]

        # comptue accuracy
        n_trial, n_feat = _x.shape
        y_pred = np.empty(n_trial)
        for train_idx, test_idx in k_fold.split(_x, _y):
            x_train = _x[train_idx, :]
            y_train = _y[train_idx]
            x_test = _x[test_idx, :]

            norm_class.fit(x_train, y_train)
            y_pred[test_idx] = norm_class.predict(x_test)

        acc_dict[label_to_rm] = accuracy_score(y_pred=y_pred, y_true =_y)
    

    weakest_link = max(acc_dict, key=acc_dict.get)

    return list(set(remaining_labels) - {weakest_link})

def find_best_combo(remaining_labels):
    # Set up combinations
    combos = list(combinations(remaining_labels, len(remaining_labels) - 1))

    scores = np.zeros(len(combos))
    matrices = []

    for scr_idx, comb in enumerate(combos):
        num_samples = 0
        for l in comb:
            num_samples += separated_data[l].shape[0]

        x_classify = np.zeros((num_samples, 21*3))
        y_subset = np.zeros(num_samples)
        sample_idx = 0 
        for l in comb:
            for t in separated_data[l]:
                x_classify[sample_idx, :] = t
                y_subset[sample_idx] = l
                sample_idx += 1

        y_pred = np.empty(x_classify.shape[0])

        for train_idx, test_idx in k_fold.split(x_classify, y_subset):
            x_train = x_classify[train_idx, :]
            y_train = y_subset[train_idx]
            x_test = x_classify[test_idx, :]

            norm_class.fit(x_train, y_train)
            y_pred[test_idx] = norm_class.predict(x_test)

        print(comb)
        print(y_subset)
        print(y_pred)
        print('-' * 10)
        
        scores[scr_idx] = accuracy_score(y_true=y_subset, y_pred=y_pred)
        matrices.append(confusion_matrix(y_pred=y_pred, y_true=y_subset))

    # Find the top score
    highest_score_idx = np.argmax(scores)
    best_combo = list(combos[highest_score_idx])
    
    return best_combo, scores, matrices

ordered_poses = np.empty(len(label_vals))
remaining_poses = np.copy(label_vals)

for n in range(len(label_vals) - 1, 1, -1):
    best = find_best_combo2(remaining_poses)
    least_distinguished = np.setdiff1d(remaining_poses, best)[0]
    print('{}: OF: {}, BEST: {}, WORST: {}'.format(n, remaining_poses, best, least_distinguished))
    remaining_poses = np.delete(remaining_poses, np.where(remaining_poses == least_distinguished))
    ordered_poses[n] = least_distinguished
    if n == 2:
        ordered_poses[0] = best[0]
        ordered_poses[1] = best[1]

print(ordered_poses.astype(int).tolist())