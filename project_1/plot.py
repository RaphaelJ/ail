from __future__ import division

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


def plot_boundary(fname, fitted_estimator, X, y, mesh_step_size=0.1, title=""):
    """Plot estimator decision boundary and scatter points

    Parameters
    ----------
    fitted_estimator : a fitted estimator

    X : array, shape (n_samples, 2)
        Input matrix

    y : array, shape (n_samples, )
        Binary classification target

    fname : str, optional (default='boundary')
        File name where the figures is saved.

    mesh_step_size : float, optional (default=0.2)
        Mesh size of the decision boundary

    title : str, optional (default="")
        Title of the graph

    """
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),
                         np.arange(y_min, y_max, mesh_step_size))

    if hasattr(fitted_estimator, "decision_function"):
        Z = fitted_estimator.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = fitted_estimator.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    plt.figure()
    plt.title(title)
    plt.xlabel("X_0")
    plt.ylabel("X_1")

    # Put the result into a color plot
    plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    plt.colorbar()

    # Plot testing point
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.savefig("%s.pdf" % fname)
    plt.close()
