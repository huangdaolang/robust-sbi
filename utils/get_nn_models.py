from typing import Callable, Optional

from torch import nn

from networks.flow import build_made, build_maf, build_nsf


def posterior_nn(
    model: str,
    z_score_theta: Optional[str] = "independent",
    z_score_x: Optional[str] = "independent",
    hidden_features: int = 50,
    num_transforms: int = 5,
    num_bins: int = 10,
    embedding_net: nn.Module = nn.Identity(),
    num_components: int = 10,
    **kwargs,
) -> Callable:
    r"""
    Returns a function that builds a density estimator for learning the posterior.

    This function will usually be used for SNPE. The returned function is to be passed
    to the inference class when using the flexible interface.

    Args:
        model: The type of density estimator that will be created. One of [`mdn`,
            `made`, `maf`, `nsf`].
        z_score_theta: Whether to z-score parameters $\theta$ before passing them into
            the network, can take one of the following:
            - `none`, or None: do not z-score.
            - `independent`: z-score each dimension independently.
            - `structured`: treat dimensions as related, therefore compute mean and std
            over the entire batch, instead of per-dimension. Should be used when each
            sample is, for example, a time series or an image.
        z_score_x: Whether to z-score simulation outputs $x$ before passing them into
            the network, same options as z_score_theta.
        hidden_features: Number of hidden features.
        num_transforms: Number of transforms when a flow is used. Only relevant if
            density estimator is a normalizing flow (i.e. currently either a `maf` or a
            `nsf`). Ignored if density estimator is a `mdn` or `made`.
        num_bins: Number of bins used for the splines in `nsf`. Ignored if density
            estimator not `nsf`.
        embedding_net: Optional embedding network for simulation outputs $x$. This
            embedding net allows to learn features from potentially high-dimensional
            simulation outputs.
        num_components: Number of mixture components for a mixture of Gaussians.
            Ignored if density estimator is not an mdn.
        kwargs: additional custom arguments passed to downstream build functions.
    """

    kwargs = dict(
        zip(
            (
                "z_score_x",
                "z_score_y",
                "hidden_features",
                "num_transforms",
                "num_bins",
                "embedding_net",
                "num_components",
            ),
            (
                z_score_theta,
                z_score_x,
                hidden_features,
                num_transforms,
                num_bins,
                embedding_net,
                num_components,
            ),
        ),
        **kwargs,
    )

    def build_fn(batch_theta, batch_x):
        if model == "made":
            return build_made(batch_x=batch_theta, batch_y=batch_x, **kwargs)
        elif model == "maf":
            return build_maf(batch_x=batch_theta, batch_y=batch_x, **kwargs)
        elif model == "nsf":
            return build_nsf(batch_x=batch_theta, batch_y=batch_x, **kwargs)
        else:
            raise NotImplementedError

    return build_fn
