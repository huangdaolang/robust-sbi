import torch


def oup(theta, n=25, N=20, var=0.1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if len(theta.shape) == 1:
        theta1 = theta[0]
        theta2 = torch.exp(theta[1])
    else:
        theta1 = theta[0, 0]
        theta2 = torch.exp(theta[0, 1])

    # noises
    T, d = 5, n+1
    dt = T/d
    Y = torch.zeros([N, n]).to(device)
    Y[:, 0] = 10
    for i in range(N):
        w = torch.normal(0., var, size=(n, 1)).to(device)

        for t in range(n-1):
            mu, sigma = theta1*(theta2 - Y[i, t])*dt, 0.5*(dt**0.5)*w[t]
            Y[i, t+1] = Y[i, t] + mu + sigma
    return Y