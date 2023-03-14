import torch

class turin():
    def __init__(self, B=4e9, Ns=801, N=100, tau0=0):
        self.B = B
        self.Ns = Ns
        self.N = N
        self.tau0 = tau0

    def __call__(self, theta, *args, **kwargs):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if len(theta.shape) == 1:
            G0 = theta[0]
            T = theta[1]
            lambda_0 = theta[2]
            sigma2_N = theta[3]
        else:
            G0 = theta[0, 0]
            T = theta[0, 1]
            lambda_0 = theta[0, 2]
            sigma2_N = theta[0, 3]

        sigma2_N = sigma2_N

        nRx = self.N

        delta_f = self.B / (self.Ns - 1)  # Frequency step size
        t_max = 1 / delta_f

        tau = torch.linspace(0, t_max, self.Ns)

        H = torch.zeros((nRx, self.Ns), dtype=torch.cfloat)

        mu_poisson = lambda_0 * t_max  # Mean of Poisson process

        for jR in range(nRx):

            n_points = int(torch.poisson(mu_poisson))  # Number of delay points sampled from Poisson process

            delays = torch.rand(n_points) * t_max  # Delays sampled from a 1-dimensional Poisson point process

            delays = torch.sort(delays)[0]

            alpha = torch.zeros(n_points,
                                dtype=torch.cfloat)  # Initialising vector of gains of length equal to the number of delay points

            sigma2 = G0 * torch.exp(-delays / T) / lambda_0 * self.B

            for l in range(n_points):
                alpha[l] = torch.normal(0, torch.sqrt(sigma2[l] / 2)) + torch.normal(0, torch.sqrt(sigma2[l] / 2)) * 1j

            H[jR, :] = torch.matmul(torch.exp(-1j * 2 * torch.pi * delta_f * (torch.ger(torch.arange(self.Ns), delays))), alpha)

        # Noise power by setting SNR
        Noise = torch.zeros((nRx, self.Ns), dtype=torch.cfloat)

        for j in range(nRx):
            normal = torch.distributions.normal.Normal(0, torch.sqrt(sigma2_N / 2))
            Noise[j, :] = normal.sample([self.Ns]) + normal.sample([self.Ns]) * 1j

        # Received signal in frequency domain

        Y = H + Noise

        y = torch.zeros(Y.shape, dtype=torch.cfloat)
        p = torch.zeros(Y.shape)
        lens = len(Y[:, 0])

        for i in range(lens):
            y[i, :] = torch.fft.ifft(Y[i, :])

            p[i, :] = torch.abs(y[i, :]) ** 2

        return 10 * torch.log10(p)