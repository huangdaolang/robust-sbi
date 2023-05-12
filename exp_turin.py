import torch

from simulators.turin import turin

from networks.summary_nets import TurinSummary
from utils.get_nn_models import *
from inference.snpe.snpe_c import SNPE_C as SNPE
from inference.base import *
from utils.torchutils import *
import pickle
import os
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(args):
    distance = args.distance
    beta = args.beta
    num_simulations = args.num_simulations
    theta_gt = args.theta
    N = args.N
    seed = args.seed

    task_name = f"{distance}_beta={beta}_num={num_simulations}_N={N}_tau0/{seed}"
    root_name = 'objects/turin/' + str(task_name)
    if not os.path.exists(root_name):
        os.makedirs(root_name)

    prior = [Uniform(1e-9*torch.ones(1).to(device), 1e-8*torch.ones(1).to(device)),
             Uniform(1e-9*torch.ones(1).to(device), 1e-8*torch.ones(1).to(device)),
             Uniform(1e7*torch.ones(1).to(device), 5e9*torch.ones(1).to(device)),
             Uniform(1e-10*torch.ones(1).to(device), 1e-9*torch.ones(1).to(device))]

    simulator, prior = prepare_for_sbi(turin(B=4e9, Ns=801, N=100, tau0=0), prior)

    sum_net = TurinSummary(input_size=1, hidden_dim=4, N=N).to(device)
    neural_posterior = posterior_nn(
        model="maf",
        embedding_net=sum_net,
        hidden_features=100,
        num_transforms=5)

    inference = SNPE(prior=prior, density_estimator=neural_posterior, device=str(device))

    x_obs = torch.tensor(np.load("data/turin_obs.npy")).float().reshape(-1, N, 801).to(device)

    if args.pre_generated_sim:
        theta = torch.tensor(np.load("data/turin_theta_2000_tau0.npy"))
        x = torch.tensor(np.load("data/turin_x_2000_tau0.npy")).reshape(num_simulations, N, 801)
    else:
        theta, x = simulate_for_sbi(simulator, prior, num_simulations=num_simulations)
    x = x.to(device)
    theta = theta.to(device)

    density_estimator = inference.append_simulations(theta, x.unsqueeze(1)).train(
        distance=distance, x_obs=x_obs, beta=beta)

    prior_new = [Uniform(1e-10*torch.ones(1).to(device), 1e-7*torch.ones(1).to(device)),
                 Uniform(1e-10*torch.ones(1).to(device), 1e-7*torch.ones(1).to(device)),
                 Uniform(1e6*torch.ones(1).to(device), 1e10*torch.ones(1).to(device)),
                 Uniform(1e-11*torch.ones(1).to(device), 1e-8*torch.ones(1).to(device))]

    simulator, prior_new = prepare_for_sbi(turin(B=4e9, Ns=801, N=100, tau0=0), prior_new)
    posterior_new = inference.build_posterior(density_estimator, prior=prior_new)
    posterior = inference.build_posterior(density_estimator, prior=prior)

    with open(root_name + "/posterior.pkl", "wb") as handle:
        pickle.dump(posterior, handle)
    with open(root_name + "/posterior_new.pkl", "wb") as handle:
        pickle.dump(posterior_new, handle)

    torch.save(sum_net, root_name + "/sum_net.pkl")
    torch.save(density_estimator, root_name + "/density_estimator.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--distance", type=str, default="none")
    parser.add_argument("--num_simulations", type=int, default=2000)
    parser.add_argument("--theta", type=list, default=[10**(-8.4), 7.8e-9, 1e9, 2.8e-10])
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--pre-generated-sim", type=bool, default=True, help="generate simulation data online or not")
    args = parser.parse_args()
    main(args)