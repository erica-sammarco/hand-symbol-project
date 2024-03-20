# hand registration: aligning hand skeletons into a common space
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

sns.set()

LIST_HAND_EDGE = np.array([[0, 1], [1, 2], [2, 3], [3, 4],
                           [0, 5], [5, 6], [6, 7], [7, 8],
                           [5, 9], [9, 10], [10, 11], [11, 12],
                           [9, 13], [13, 14], [14, 15], [15, 16],
                           [13, 17], [17, 18], [18, 19], [19, 20],
                           [17, 0]])


# 21 hand landmarks whose xyz coordinates (3) are given
def reshape_to_landmark(x, shape=(21, 3)):
    """ converts 1st dim (size 63) to 21 x 3, one row per landmark """
    assert x.shape[0] == np.prod(shape)
    return x.reshape(*shape, -1)


def reshape_to_sklearn(x, shape_from=(21, 3)):
    assert x.shape[:2] == shape_from
    return x.reshape(-1, *x.shape[2:])


def plot_hand(x_posisitons, y_positions, **kwargs):
    # Create a NetworkX graph
    G = nx.Graph()

    # Add nodes to the graph with (x, y) positions
    for i, (x, y) in enumerate(zip(x_posisitons, y_positions)):
        G.add_node(i, pos=(x, y))

    # Get positions of nodes
    node_positions = nx.get_node_attributes(G, 'pos')

    G.add_edges_from(LIST_HAND_EDGE)

    # Plot the graph
    nx.draw(G, pos=node_positions, with_labels=False, node_size=0,
            font_size=12, **kwargs)
    plt.plot()


class HandRegister:
    """

    Attributes:
        list_stationary (list): index of all points in hand which should be
            stationary under registration.
    """

    def __init__(self, list_stationary=(0, 5, 9, 13, 17)):
        self.list_stationary = list_stationary

        self.x_template = None

    def fit(self, x):
        """ build template position we seek to project to

        Args:
            x (np.array): (63, n_trials) positions
        """
        # reshape into (21, 3, n_trials), easier to access per hand landmark
        x = reshape_to_landmark(x)

        # discard all but stationary points
        x = x[self.list_stationary, :, :]

        # averaging across all trials to build template
        self.x_template = x.mean(axis=2)

    def transform(self, x):
        """ transforms each trial into a common space

        Args:
            x (np.array): (63, n_trials) positions

        Returns:
            x_out (np.array): (63, n_trials) positions (in common space)
        """
        # reshape into (21, 3, n_trials), easier to access per hand landmark
        x = reshape_to_landmark(x)

        x_out = np.empty_like(x)
        for trial_idx in range(x.shape[2]):
            # adding bias col of 1s (offset)
            _x = x[:, :, trial_idx]
            _x = np.hstack((np.ones((_x.shape[0], 1)), _x))

            # compute min MSE transformation back to template space
            pinv = np.linalg.pinv(_x[self.list_stationary, :])
            a = pinv @ self.x_template

            # apply and store
            x_out[:, :, trial_idx] = _x @ a

        return reshape_to_sklearn(x_out)
