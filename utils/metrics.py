# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import Any, Dict, Optional

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from torch import Tensor
from scipy.stats import gaussian_kde
from scipy.spatial import cKDTree as KDTree

def c2st(
    X: Tensor,
    Y: Tensor,
    seed: int = 1,
    n_folds: int = 5,
    metric: str = "accuracy",
    classifier: str = "rf",
) -> Tensor:
    """
    Return accuracy of classifier trained to distinguish samples from supposedly
    two distributions <X> and <Y>. For details on the method, see [1,2].
    If the returned accuracy is 0.5, <X> and <Y> are considered to be from the
    same generating PDF, i.e. they can not be differentiated.
    If the returned accuracy is around 1., <X> and <Y> are considered to be from
    two different generating PDFs.

    Training of the classifier with N-fold cross-validation [3] using sklearn.
    By default, a `RandomForestClassifier` by from `sklearn.ensemble` is used
    (<classifier> = 'rf'). Alternatively, a multi-layer perceptron is available
    (<classifier> = 'mlp'). For a small study on the pros and cons for this
    choice see [4]. Before both samples are ingested, they are normalized (z scored)
    under the assumption that each dimension in X follows a normal distribution, i.e.
    the mean(X) is subtracted from X and this difference is divided by std(X)
    for every dimension.

    If you need a more flexible interface which is able to take a sklearn
    compatible classifier and more, see the `c2st_` method in this module.

    Args:
        X: Samples from one distribution.
        Y: Samples from another distribution.
        seed: Seed for the sklearn classifier and the KFold cross-validation
        n_folds: Number of folds to use
        metric: sklearn compliant metric to use for the scoring parameter of cross_val_score
        classifier: classification architecture to use, possible values: 'rf' or 'mlp'

    Return:
        torch.tensor containing the mean accuracy score over the test sets
        from cross-validation

    Example:
    ``` py
    > c2st(X,Y)
    [0.51904464] #X and Y likely come from the same PDF or ensemble
    > c2st(P,Q)
    [0.998456] #P and Q likely come from two different PDFs or ensembles
    ```

    References:
        [1]: http://arxiv.org/abs/1610.06545
        [2]: https://www.osti.gov/biblio/826696/
        [3]: https://scikit-learn.org/stable/modules/cross_validation.html
        [4]: https://github.com/psteinb/c2st/
    """

    # the default configuration
    clf_class = RandomForestClassifier
    clf_kwargs = {}

    if "mlp" in classifier.lower():
        ndim = X.shape[-1]
        clf_class = MLPClassifier
        clf_kwargs = {
            "activation": "relu",
            "hidden_layer_sizes": (10 * ndim, 10 * ndim),
            "max_iter": 1000,
            "solver": "adam",
            "early_stopping": True,
            "n_iter_no_change": 50,
        }

    noise_scale = None
    z_score = True
    verbosity = 0

    scores_ = c2st_scores(
        X,
        Y,
        seed=seed,
        n_folds=n_folds,
        metric=metric,
        z_score=z_score,
        noise_scale=noise_scale,
        verbosity=verbosity,
        clf_class=clf_class,
        clf_kwargs=clf_kwargs,
    )

    # TODO: unclear why np.asarray needs to be used here
    scores = np.asarray(np.mean(scores_)).astype(np.float32)
    value = torch.from_numpy(np.atleast_1d(scores))
    return value


def c2st_scores(
    X: Tensor,
    Y: Tensor,
    seed: int = 1,
    n_folds: int = 5,
    metric: str = "accuracy",
    z_score: bool = True,
    noise_scale: Optional[float] = None,
    verbosity: int = 0,
    clf_class: Any = RandomForestClassifier,
    clf_kwargs: Dict[str, Any] = {},
) -> Tensor:
    """
    Return accuracy of classifier trained to distinguish samples from supposedly
    two distributions <X> and <Y>. For details on the method, see [1,2].
    If the returned accuracy is 0.5, <X> and <Y> are considered to be from the
    same generating PDF, i.e. they can not be differentiated.
    If the returned accuracy is around 1., <X> and <Y> are considered to be from
    two different generating PDFs.

    This function performs training of the classifier with N-fold cross-validation [3] using sklearn.
    By default, a `RandomForestClassifier` by from `sklearn.ensemble` is used which
    is recommended based on the study performed in [4].
    This can be changed using <clf_class>. This class is used in the following
    fashion:

    ``` py
    clf = clf_class(random_state=seed, **clf_kwargs)
    #...
    scores = cross_val_score(
        clf, data, target, cv=shuffle, scoring=scoring, verbose=verbosity
    )
    ```
    Further configuration of the classifier can be performed using <clf_kwargs>.
    If you like to provide a custom class for training, it has to satisfy the
    internal requirements of `sklearn.model_selection.cross_val_score`.

    Args:
        X: Samples from one distribution.
        Y: Samples from another distribution.
        seed: Seed for the sklearn classifier and the KFold cross validation
        n_folds: Number of folds to use for cross validation
        metric: sklearn compliant metric to use for the scoring parameter of cross_val_score
        z_score: Z-scoring using X, i.e. mean and std deviation of X is used to normalize Y, i.e. Y=(Y - mean)/std
        noise_scale: If passed, will add Gaussian noise with standard deviation <noise_scale> to samples of X and of Y
        verbosity: control the verbosity of sklearn.model_selection.cross_val_score
        clf_class: a scikit-learn classifier class
        clf_kwargs: key-value arguments dictionary to the class specified by clf_class, e.g. sklearn.ensemble.RandomForestClassifier

    Return:
        np.ndarray containing the calculated <metric> scores over the test set
        folds from cross-validation

    Example:
    ``` py
    > c2st_scores(X,Y)
    [0.51904464,0.5309201,0.4959452,0.5487709,0.50682926]
    #X and Y likely come from the same PDF or ensemble
    > c2st_scores(P,Q)
    [0.998456,0.9982912,0.9980476,0.9980488,0.99805826]
    #P and Q likely come from two different PDFs or ensembles
    ```

    References:
        [1]: http://arxiv.org/abs/1610.06545
        [2]: https://www.osti.gov/biblio/826696/
        [3]: https://scikit-learn.org/stable/modules/cross_validation.html
        [4]: https://github.com/psteinb/c2st/
    """
    if z_score:
        X_mean = torch.mean(X, dim=0)
        X_std = torch.std(X, dim=0)
        X = (X - X_mean) / X_std
        Y = (Y - X_mean) / X_std

    if noise_scale is not None:
        X += noise_scale * torch.randn(X.shape)
        Y += noise_scale * torch.randn(Y.shape)

    X = X.cpu().numpy()
    Y = Y.cpu().numpy()

    clf = clf_class(random_state=seed, **clf_kwargs)

    # prepare data
    data = np.concatenate((X, Y))
    # labels
    target = np.concatenate((np.zeros((X.shape[0],)), np.ones((Y.shape[0],))))

    shuffle = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(
        clf, data, target, cv=shuffle, scoring=metric, verbose=verbosity
    )

    return scores


def unbiased_mmd_squared(x, y):
    nx, ny = x.shape[0], y.shape[0]

    def f(a, b, diag=False):
        if diag:
            return torch.sum((a[None, ...] - b[:, None, :]) ** 2, dim=-1).reshape(-1)
        else:
            m, n = a.shape[0], b.shape[0]
            ix = torch.tril_indices(m, n, offset=-1)
            return torch.sum(
                (a[None, ...] - b[:, None, :]) ** 2, dim=-1, keepdim=False
            )[ix[0, :], ix[1, :]].reshape(-1)

    xx = f(x, x)
    xy = f(x, y, diag=True)
    yy = f(y, y)

    scale = torch.median(torch.sqrt(torch.cat((xx, xy, yy))))
    c = -0.5 / (scale**2)

    k = lambda a: torch.sum(torch.exp(c * a))

    kxx = k(xx) / (nx * (nx - 1))
    kxy = k(xy) / (nx * ny)
    kyy = k(yy) / (ny * (ny - 1))
    del xx, xy, yy

    mmd_square = 2 * (kxx + kyy - kxy)
    del kxx, kxy, kyy

    return mmd_square


def biased_mmd(x, y):
    nx, ny = x.shape[0], y.shape[0]

    def f(a, b):
        return torch.sum((a[None, ...] - b[:, None, :]) ** 2, dim=-1).reshape(-1)

    xx = f(x, x)
    xy = f(x, y)
    yy = f(y, y)

    scale = torch.median(torch.sqrt(torch.cat((xx, xy, yy))))
    c = -0.5 / (scale**2)

    k = lambda a: torch.sum(torch.exp(c * a))

    kxx = k(xx) / nx**2
    del xx
    kxy = k(xy) / (nx * ny)
    del xy
    kyy = k(yy) / ny**2
    del yy

    mmd_square = kxx - 2 * kxy + kyy
    del kxx, kxy, kyy

    return torch.sqrt(mmd_square)


def biased_mmd_hypothesis_test(x, y, alpha=0.05):
    assert x.shape[0] == y.shape[0]
    mmd_biased = biased_mmd(x, y).item()
    threshold = np.sqrt(2 / x.shape[0]) * (1 + np.sqrt(-2 * np.log(alpha)))

    return mmd_biased, threshold


def unbiased_mmd_squared_hypothesis_test(x, y, alpha=0.05):
    assert x.shape[0] == y.shape[0]
    mmd_square_unbiased = unbiased_mmd_squared(x, y).item()
    threshold = (4 / np.sqrt(x.shape[0])) * np.sqrt(-np.log(alpha))

    return mmd_square_unbiased, threshold


def _test():
    n = 2500
    x, y = torch.randn(n, 5), torch.randn(n, 5)
    print(unbiased_mmd_squared(x, y), biased_mmd(x, y))
    # mmd(x, y), sq_maximum_mean_discrepancy(tensor2numpy(x), tensor2numpy(y))
    # mmd_hypothesis_test(x, y, alpha=0.0001)
    # unbiased_mmd_squared_hypothesis_test(x, y)


def MMD_unweighted(x, y, lengthscale):
    """ Approximates the squared MMD between samples x_i ~ P and y_i ~ Q
    """

    m = x.shape[0]
    n = y.shape[0]

    z = torch.cat((x, y), dim=0)

    K = kernel_matrix(z, z, lengthscale)

    kxx = K[0:m, 0:m]
    kyy = K[m:(m + n), m:(m + n)]
    kxy = K[0:m, m:(m + n)]

    return (1 / m ** 2) * torch.sum(kxx) - (2 / (m * n)) * torch.sum(kxy) + (1 / n ** 2) * torch.sum(kyy)


def median_heuristic(y):
    a = torch.cdist(y, y)**2
    return torch.sqrt(torch.median(a / 2))


def kernel_matrix(x, y, l):
    d = torch.cdist(x, y)**2

    kernel = torch.exp(-(1 / (2 * l ** 2)) * d)

    return kernel


def RMSE(gt, samples, p=1):
    if p == 1:
        dist = torch.mean(torch.abs(gt-samples))
    elif p == 2:
        dist = torch.sqrt(torch.mean((gt-samples)**2))
    elif p == 3:
        dist = torch.nn.functional.pairwise_distance(gt, samples, p=2).mean()
    else:
        raise NotImplementedError
    return dist


def KLdivergence(x, y):
    """Compute the Kullback-Leibler divergence between two multivariate samples.

    Parameters
    ----------
    x : 2D array (n,d)
    Samples from distribution P, which typically represents the true
    distribution.
    y : 2D array (m,d)
    Samples from distribution Q, which typically represents the approximate
    distribution.

    Returns
    -------
    out : float
    The estimated Kullback-Leibler divergence D(P||Q).

    References
    ----------
    Pérez-Cruz, F. Kullback-Leibler divergence estimation of
    continuous distributions IEEE International Symposium on Information
    Theory, 2008.
    """

    # Check the dimensions are consistent
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    n,d = x.shape
    m,dy = y.shape

    assert(d == dy)


    # Build a KD tree representation of the samples and find the nearest neighbour
    # of each point in x.
    xtree = KDTree(x)
    ytree = KDTree(y)

    # Get the first two nearest neighbours for x, since the closest one is the
    # sample itself.
    r = xtree.query(x, k=2, eps=.01, p=2)[0][:,1]
    s = ytree.query(x, k=1, eps=.01, p=2)[0]

    # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
    # on the first term of the right hand side.
    return -np.log(r/s).sum() * d / n + np.log(m / (n - 1.))


def main():
    _test()


if __name__ == "__main__":
    main()
