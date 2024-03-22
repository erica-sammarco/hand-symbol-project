import numpy as np
from scipy.stats import multivariate_normal


class NormClass:
    """ assumes additive white gaussian noise on some template per target

    (we subtract the average hand pose and build a model on errors)
    """
    pass

    def __init__(self):
        self.mv_norm_list = None
        self.class_label = None

    def fit(self, x, y):
        """ template is average across all trials """
        self.class_label = np.unique(y)
        self.mv_norm_list = list()

        # compute mu
        mu_list = list()
        sample_centered = list()
        for _y in self.class_label:
            _x = x[_y == y, :]
            mu = _x.mean(axis=0)
            mu_list.append(mu)
            sample_centered.append(_x - mu)

        # compute cov
        error = np.vstack(sample_centered).flatten()
        var = (error ** 2).sum() / (error.size - 1)
        cov = np.eye(x.shape[1]) * var

        # build mv_norm
        self.mv_norm_list = [multivariate_normal(mean=mu, cov=cov)
                             for mu in mu_list]

    def predict(self, *args, **kwargs):
        """ predict class with min sum of squared error (all have same var) """
        log_prob = self.predict_log_prob(*args, **kwargs)
        return log_prob.argmax(axis=1)

    def predict_log_prob(self, x):
        n_sample, n_feat = x.shape
        n_class = len(self.class_label)
        log_like = np.zeros((n_sample, n_class))
        for class_idx in range(n_class):
            log_like[:, class_idx] = self.mv_norm_list[class_idx].logpdf(x)

        return log_like

    def predict_apost(self, *args, **kwargs):
        """ a posteriori """
        log_like = self.predict_log_prob(*args, **kwargs)
        log_like -= log_like.max(axis=1)[:, np.newaxis]
        prob = np.exp(log_like)
        prob /= prob.sum(axis=1)[:, np.newaxis]
        return prob
