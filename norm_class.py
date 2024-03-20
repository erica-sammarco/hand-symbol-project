from experiment.parse import *


class NormClass:
    """ assumes additive white gaussian noise on some template per target

    (we subtract the average hand pose and build a model on errors)
    """
    pass

    def __init__(self):
        self.template = None
        self.class_label = None

    def train(self, x, y):
        """ template is average across all trials """
        self.class_label = np.unique(y)
        self.template = np.zeros((x.shape[0], self.class_label.size))
        error = np.zeros_like(x)
        for idx, _y in enumerate(self.class_label):
            _x = x[:, _y == y]
            self.template[:, idx] = _x.mean(axis=1)
            error[:, _y == y] = self.template[:, idx][:, np.newaxis] - _x

        # estimate variance (needed to compute absolute likelihoods)
        # self.var = (error ** 2).sum() / (error.size - 1)

    def predict(self, x):
        """ predict class with min sum of squared error (all have same var) """
        error = self._get_error(x)
        y_pred_idx = np.argmin((error ** 2).sum(axis=0), axis=1)
        return self.class_label[y_pred_idx]

    def _get_error(self, x):
        """ for each sample & target class, compute sum of squared error """
        n_feature, n_sample = x.shape
        error = np.zeros((n_feature, n_sample, self.class_label.size))
        for class_idx in range(self.class_label.size):
            # sum of squared errors
            temp = self.template[:, class_idx]
            error[:, :, class_idx] = (x - temp[:, np.newaxis])

        return error

    def predict_proba(self, x):
        error = self._get_error(x)

        # not normalized log likelihood
        log_like = -np.log(error ** 2).sum(axis=0)

        # subtract max from likelihoods (numerical precision if many features)
        log_like -= log_like.max(axis=1)[:, np.newaxis]

        # out of log space & normalize
        prob = np.exp(log_like)
        prob /= prob.sum(axis=1)[:, np.newaxis]

        return prob
