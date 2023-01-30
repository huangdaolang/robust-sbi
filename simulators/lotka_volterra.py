import numpy as np
import torch
from scipy.integrate import odeint


def _deriv(x, t, alpha, beta, gamma, delta):
    """Helper function for scipy.integrate.odeint."""

    X, Y = x
    dX = alpha * X - beta * X * Y
    dY = -gamma * Y + delta * X * Y
    return dX, dY


def lotka_volterra(theta, X0=30, Y0=1, T=20, flatten=True):
    """Runs a Lotka-Volterra simulation for T time steps and returns `subsample` evenly spaced
    points from the simulated trajectory, given contact parameters `theta`.
    See https://arxiv.org/pdf/2101.04653.pdf, Benchmark Task T.10.

    Parameters
    ----------
    theta       : np.ndarray of shape (2,)
        The 2-dimensional vector of disease parameters.
    X0          : float, optional, default: 30
        Initial number of prey species.
    Y0          : float, optional, default: 1
        Initial number of predator species.
    T           : T, optional, default: 20
        The duration (time horizon) of the simulation.
    subsample   : int or None, optional, default: 10
        The number of evenly spaced time points to return. If None,
        no subsampling will be performed and all T timepoints will be returned.
    flatten     : bool, optional, default: True
        A flag to indicate whather a 1D (`flatten=True`) or a 2D (`flatten=False`)
        representation of the simulated data is returned.
    rng         : np.random.Generator or None, default: None
        An optional random number generator to use.
    Returns
    -------
    x : np.ndarray of shape (subsample, 2) or (subsample*2,) if `subsample is not None`,
        otherwise shape (T, 2) or (T*2,) if `subsample is None`.
        The time series of simulated predator and pray populations
    """
    nSamples = 100

    theta = theta[0].detach().numpy()

    Y = torch.zeros(size=(nSamples, T * 2))
    # Use default RNG, if None specified
    for i in range(nSamples):
        rng = np.random.default_rng()

        # Create vector (list) of initial conditions
        x0 = X0, Y0

        # Unpack parameter vector into scalars
        alpha, beta, gamma, delta = theta

        # Prepate time vector between 0 and T of length T
        t_vec = np.linspace(0, T, T)

        # Integrate using scipy and retain only infected (2-nd dimension)
        pp = odeint(_deriv, x0, t_vec, args=(alpha, beta, gamma, delta))

        # Ensure minimum count is 0, which will later pass by log(0 + 1)
        pp[pp < 0] = 0

        # Add noise, decide whether to flatten and return
        x = rng.lognormal(np.log1p(pp), sigma=0.1)
        if flatten:
            Y[i] = torch.tensor(x.flatten())
    return Y