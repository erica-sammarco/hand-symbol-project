from parse import *


class NormClass:
    """ assumes additive white gaussian noise on some template per target

    (we subtract the average hand pose and build a model on errors)
    """
    pass

    def __init__(self):
        pass

    def train(self, x, y):
        pass

    def predict(self, x):
        pass

    def predict_proba(self, x):
        pass


if __name__ == '__main__':
    import pathlib

    path_input = pathlib.Path('./experiment/prompt-results/jtdawc_3.txt')
    with open(path_input, 'r') as f:
        s = f.read()

    d = parse(s)

    x = np.stack([Landmark.to_numpy(lmk_list) for lmk_list in d['normalized']])
