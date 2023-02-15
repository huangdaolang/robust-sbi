import numpy as np
import random
import torch


def sparsity(obs):
    obs_cont = torch.clone(obs).to(obs.device)
    num_cont = int(obs.shape[0])

    for i in range(num_cont):
        # start = random.randint(0, 80)
        # obs_cont[i, start:start+20] = 0
        obs_cont[i, 20:40] = 0

    # obs_cont = torch.tensor(obs_cont)
    return obs_cont


def magnitude(obs, degree=0.2, var=15):
    obs_cont = torch.clone(obs).to(obs.device)

    num_total = int(obs.shape[0])
    num_cont = int(num_total * degree)
    index_list = [int(i) for i in range(num_total)]
    random.shuffle(index_list)

    for i in range(num_cont):
        obs_cont[index_list[i]] += torch.abs(torch.randn(100).to(obs.device) * var)

    return obs_cont


def magnitude_sigma(obs, var=15, length=100, N=100):
    obs_cont = torch.clone(obs).to(obs.device)

    obs_cont += torch.abs(torch.randn(N, length).to(obs.device) * var)

    # obs_cont = torch.tensor(obs_cont)
    return obs_cont
