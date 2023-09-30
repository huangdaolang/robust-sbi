import torch


class oup():
    def __init__(self, n=25, N=20, var=0.1):
        self.n = n
        self.N = N
        self.var = var

    def __call__(self, theta, *args, **kwargs):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if len(theta.shape) == 1:
            theta1 = theta[0]
            theta2 = torch.exp(theta[1])
        else:
            theta1 = theta[0, 0]
            theta2 = torch.exp(theta[0, 1])

        # noises
        T, d = 5, self.n+1
        dt = T/d
        Y = torch.zeros([self.N, self.n]).to(device)
        Y[:, 0] = 10
        for i in range(self.N):
            w = torch.normal(0., self.var, size=(self.n, 1)).to(device)

            for t in range(self.n-1):
                mu, sigma = theta1*(theta2 - Y[i, t])*dt, 0.5*(dt**0.5)*w[t]
                Y[i, t+1] = Y[i, t] + mu + sigma
        return Y

    def get_name(self):
        return "oup"