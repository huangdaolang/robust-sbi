import torch
import numpy as np


class ricker():
    def __init__(self, N=20):
        self.N = N

    def __call__(self, theta, *args, **kwargs):
        if len(theta.shape) == 1:
            logr = theta[0]
            phi = theta[1]
        else:
            logr = theta[0, 0]
            phi = theta[0, 1]
        if phi < 0:
            phi = 0.001
        sigma = 0.3

        N0 = 1
        T = 100
        nSamples = self.N

        Y = torch.zeros(size=(nSamples, T))

        for i in range(nSamples):
            et = torch.randn(T, )
            Nt = torch.zeros(size=(T + 1,))

            Nt[0] = N0

            for t in range(1, T + 1):
                Nt[t] = torch.exp(logr + torch.log(Nt[t - 1]) - Nt[t - 1] + sigma * et[t - 1])
                Y[i, t - 1] = torch.poisson(phi * Nt[t])
        return Y

    def get_name(self):
        return "ricker"
