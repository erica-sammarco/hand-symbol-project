from itertools import product

import numpy as np
from sklearn.metrics import accuracy_score

from norm_class import NormClass


def test_norm_class():
    # generate dummy data
    n_feat, n_sample = 15, 100
    n_class = 3
    # each (snr, acc) pair gives lower bound for accuracy at every snr
    snr_acc_list = [(.1, .5),
                    (3, .95),
                    (10, 1)]
    for seed, (snr, acc_exp) in product(range(3), snr_acc_list):
        rng = np.random.default_rng(seed=seed)

        # build ground truth targets
        template_true = rng.standard_normal((n_class, n_feat))

        # sample x
        x = rng.standard_normal((n_sample, n_feat)) * (1 / snr) ** .5
        y_true = np.arange(n_sample) % n_class
        for class_idx in range(n_class):
            x[y_true == class_idx, :] += template_true[class_idx, :]

        # at given snr, we should have at least acc_exp accuracy
        norm_class = NormClass()
        norm_class.train(x, y_true)
        y_pred = norm_class.predict(x)
        assert accuracy_score(y_true=y_true, y_pred=y_pred) >= acc_exp

        # ensure probabilities are consistent with point estimates above
        prob = norm_class.predict_apost(x)
        assert np.allclose(1, prob.sum(axis=1))
        assert np.allclose(np.argmax(prob, axis=1), y_pred)
