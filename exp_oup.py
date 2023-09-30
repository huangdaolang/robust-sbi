import torch

from simulators.oup import oup

from networks.summary_nets import OUPSummary
from utils.get_nn_models import *
from inference.snpe.snpe_c import SNPE_C as SNPE
from inference.base import *
from utils.torchutils import *
import pickle
import os
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(args):
    var = args.var
    distance = args.distance
    beta = args.beta
    num_simulations = args.num_simulations
    theta_gt = args.theta
    N = args.N
    degree = args.degree
    n_corrupted = int(N * degree)
    n_normal = int(N - n_corrupted)

    task_name = f"degree={degree}_var={var}_{distance}_beta={beta}_theta={theta_gt}_num={num_simulations}/{str(args.seed)}"
    root_name = 'objects/oup_final/' + str(task_name)
    if not os.path.exists(root_name):
        os.makedirs(root_name)

    prior = [Uniform(torch.zeros(1).to(device), 2 * torch.ones(1).to(device)),
             Uniform(-2 * torch.ones(1).to(device), 2 * torch.ones(1).to(device))]
    simulator, prior = prepare_for_sbi(oup(N=N, var=var), prior)

    sum_net = OUPSummary(input_size=1, hidden_dim=2, N=N).to(device)
    neural_posterior = posterior_nn(
        model="maf",
        embedding_net=sum_net,
        hidden_features=20,
        num_transforms=3)

    inference = SNPE(prior=prior, density_estimator=neural_posterior, device=str(device))

    if args.pre_generated_obs:
        obs_cont = torch.tensor(np.load(f"data/oup_obs_{int(degree * 10)}.npy"))
    else:
        theta_gt = torch.tensor(theta_gt)
        theta_cont = torch.Tensor([-0.5, 1])
        oup_obs = oup(var=var, N=N)
        obs = oup_obs(theta_gt).to(device)
        oup_obs_cont = oup(var=1, N=N)
        obs_2 = oup_obs_cont(theta_cont).to(device)
        obs_cont = torch.cat([obs[:n_normal], obs_2[:n_corrupted]], dim=0).reshape(-1, N, 25)

    if args.pre_generated_sim:
        theta = torch.tensor(np.load("data/oup_theta_1000.npy"))
        x = torch.tensor(np.load("data/oup_x_1000.npy")).reshape(num_simulations, N, 25)
    else:
        theta, x = simulate_for_sbi(simulator, prior, num_simulations=num_simulations)

    x = x.reshape(num_simulations, N, 25).to(device)
    theta = theta.to(device)
    density_estimator = inference.append_simulations(theta, x.unsqueeze(1)).train(
        distance=distance, x_obs=obs_cont, beta=beta)

    # increase the prior range in case we can't generate thetas for mis-specified observation
    prior_new = [Uniform(-20 * torch.ones(1), 20 * torch.ones(1)),
                 Uniform(-20 * torch.ones(1), 20 * torch.ones(1))]
    simulator, prior_new = prepare_for_sbi(oup(N=N, var=var), prior_new)
    posterior = inference.build_posterior(density_estimator, prior=prior_new)

    with open(root_name + "/posterior.pkl", "wb") as handle:
        pickle.dump(posterior, handle)

    torch.save(sum_net, root_name + "/sum_net.pkl")
    torch.save(density_estimator, root_name + "/density_estimator.pkl")

    if args.keep_inference:
        with open(root_name + "/inference.pkl", "wb") as handle:
            pickle.dump(inference, handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=1.0, help="regularization weight")
    parser.add_argument("--degree", type=float, default=0.2, help="degree of mis-specification")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--distance", type=str, default="mmd", choices=["euclidean", "none", "mmd"])
    parser.add_argument("--num_simulations", type=int, default=1000, help="number of simulations")
    parser.add_argument("--theta", type=list, default=[0.5, 1.0], help="ground truth theta")
    parser.add_argument("--N", type=int, default=100, help="Number of realizations for each set of theta")
    parser.add_argument("--pre-generated-sim", action="store_true", help="generate simulation data online or not")
    parser.add_argument("--pre-generated-obs", action="store_true", help="generate observation data online or not")
    parser.add_argument("--keep-inference", action="store_true", help="save inference model or not")
    args = parser.parse_args()
    main(args)
