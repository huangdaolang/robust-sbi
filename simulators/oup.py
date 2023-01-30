import torch


def oup(theta, n=50, N=100):
    if len(theta.shape) == 1:
        theta1 = theta[0]
        theta2 = torch.exp(theta[1])
    else:
        theta1 = theta[0, 0]
        theta2 = torch.exp(theta[0, 1])

    # noises
    T, d = 10.0, n+1
    dt = T/d
    Y = torch.zeros([N, n])
    Y[:, 0] = 10
    for i in range(N):
        w = torch.normal(0., 1., size=(n,1))

        for t in range(n-1):
            mu, sigma = theta1*(theta2 - Y[i, t])*dt, 0.5*(dt**0.5)*w[t]
            Y[i, t+1] = Y[i, t] + mu + sigma
    return Y