import numpy as np
import torch
import corruption


x = torch.tensor(np.load("../data/x_10.npy")).reshape(10, 100, 100)
theta = torch.tensor(np.load("../data/theta_10.npy"))
num_simulations = theta.shape[0]
sigmas = torch.tensor(np.random.uniform(5, 20, size=[num_simulations, 5]))

x_corrupted = torch.zeros([num_simulations, 6, 100, 100])

for i in range(num_simulations):
    x_corrupted[i, 0] = x[i]
    for j in range(1, 6):
        x_corrupted[i, j] = corruption.magnitude_sigma(x[i], var=sigmas[i, j-1])

np.save("../x_10_corrupted.npy", x_corrupted.cpu().detach().numpy())




