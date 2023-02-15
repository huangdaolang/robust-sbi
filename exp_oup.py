import torch

from simulators.oup import oup

from networks.summary_nets import OUPSummary
from utils.get_nn_models import *
from inference.snpe.snpe_c import SNPE_C as SNPE
from inference.base import *
from utils.sbiutils import *
from utils.torchutils import *
import pickle
import os
import argparse
import utils.corruption as corruption

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

    task_name = "degree={degree}_var={var}_{distance}_beta={beta}_theta={theta}_num={nums}_N={N}/{seed}".format(var=var,
                                                                               distance=distance,
                                                                               beta=beta,
                                                                               theta=theta_gt,
                                                                               seed=str(args.seed),
                                                                               nums=num_simulations,
                                                                               N=N,
                                                                               degree=degree)
    root_name = 'objects/oup/' + str(task_name)
    if not os.path.exists(root_name):
        os.makedirs(root_name)

    prior = [Uniform(- torch.ones(1).to(device), 2 * torch.ones(1).to(device)),
             Uniform(-2 * torch.ones(1).to(device), 2 * torch.ones(1).to(device))]
    simulator, prior = prepare_for_sbi(oup(N=N), prior)

    sum_net = OUPSummary(input_size=1, hidden_dim=2, N=N).to(device)
    neural_posterior = posterior_nn(
        model="maf",
        embedding_net=sum_net,
        hidden_features=20,
        num_transforms=3)

    inference = SNPE(prior=prior, density_estimator=neural_posterior, device=str(device))

    theta_gt = torch.tensor(theta_gt)
    theta_cont = torch.Tensor([-0.5, 1])
    oup_obs = oup(var=var, N=N)
    obs = oup_obs(theta_gt).to(device)
    oup_obs_cont = oup(var=1, N=N)
    obs_2 = oup_obs_cont(theta_cont).to(device)
    obs_cont = torch.cat([obs[:n_normal], obs_2[:n_corrupted]], dim=0).reshape(-1, N, 25)

    theta, x = simulate_for_sbi(simulator, prior, num_simulations=num_simulations)

    x = x.reshape(num_simulations, N, 25).to(device)
    theta = theta.to(device)
    density_estimator = inference.append_simulations(theta, x.unsqueeze(1)).train(
        corrupt_data_training=distance, x_obs=obs_cont)

    prior_new = [Uniform(-1 * torch.ones(1), 4 * torch.ones(1)),
                 Uniform(-4 * torch.zeros(1), 4 * torch.ones(1))]
    simulator, prior_new = prepare_for_sbi(oup, prior_new)
    posterior = inference.build_posterior(density_estimator, prior=prior_new)

    with open(root_name + "/posterior.pkl", "wb") as handle:
        pickle.dump(posterior, handle)

    torch.save(sum_net, root_name + "/sum_net.pkl")
    torch.save(density_estimator, root_name + "/density_estimator.pkl")

    # with open(root_name + "/inference.pkl", "wb") as handle:
    #     pickle.dump(inference, handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--distance", type=str, default="mmd")
    parser.add_argument("--num_simulations", type=int, default=1000)
    parser.add_argument("--var", type=float, default=1)
    parser.add_argument("--theta", type=list, default=[0.5, 1.0])
    parser.add_argument("--N", type=int, default=100)
    args = parser.parse_args()
    main(args)
