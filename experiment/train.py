import numpy as np
import pathlib
from norm_class import NormClass
import parse
from register import HandRegister
from sklearn.metrics import accuracy_score


def process_file_data(filepath):
    with open(filepath, 'r') as f:
        lmk, lmk_world, trigger = parse.parse(f.read())

    # flatten 21 x 3 features into a vector of length 63
    lmk = lmk.reshape((-1, trigger.size))

    # cut stream of data into trials
    idx_trial_start, y, trial_length = parse.split_trial(trigger=trigger)
    min_trial_length = min(trial_length)
    x_raw = np.stack(
        [lmk[:, idx: idx + min_trial_length] for idx in idx_trial_start], axis=-1)

    # dimensions: features (21 x 3), trial
    x_raw = x_raw.mean(axis=1)

    hand_reg = HandRegister()
    hand_reg.fit(x_raw)
    x = hand_reg.transform(x_raw)

    x_sklearn = x.T

    return x_sklearn, y, trial_length


def get_trained_classifier(participant_id):
    # load data
    folder_key = pathlib.Path(__file__).parent / 'calibration-results'
    input_file = '{}/{}.txt'.format(folder_key, participant_id)
    x_sklearn, y, trial_length = process_file_data(input_file)

    clf = NormClass()
    clf.fit(x_sklearn, y)

    return clf

if __name__ == '__main__':
    participant_id = "EOx0vb"
    
    clf = get_trained_classifier(participant_id)

    folder_key = pathlib.Path(__file__).parent / 'prompt-results'
    input_file = '{}/{}_1.txt'.format(folder_key, participant_id)
    x, y, trial_length = process_file_data(input_file)
    print(x.shape)
    print(y.shape)
    y_pred = clf.predict(x)

    print(y)
    print(y_pred)

    print(accuracy_score(y_pred=y_pred, y_true=y))