import numpy as np
import parse
from register import HandRegister
from sklearn.neighbors import KNeighborsClassifier

def process_file_data(filepath):
    with open(filepath, 'r') as f: 
        lmk, lmk_world, trigger = parse.parse(f.read())
    
    # flatten 21 x 3 features into a vector of length 63
    lmk = lmk.reshape((-1, trigger.size))

    # cut stream of data into trials
    idx_trial_start, y, trial_length = parse.split_trial(trigger=trigger)
    x_raw = np.stack([lmk[:, idx: idx + trial_length] for idx in idx_trial_start], axis=-1)

    # dimensions: features (21 x 3), trial
    x_raw = x_raw.mean(axis=1)

    hand_reg = HandRegister()
    hand_reg.fit(x_raw)
    x = hand_reg.transform(x_raw)

    x_sklearn = x.T

    return x_sklearn, y

def get_trained_classifier(participant_id):
    # load data
    input_file = './calibration-results/{}.txt'.format(participant_id)
    x_sklearn, y = process_file_data(input_file)

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(x_sklearn, y)

    return neigh
