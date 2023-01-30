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

    task_name = "var={var}_{distance}_beta={beta}_theta={theta}/{seed}".format(var=var,
                                                                               distance=distance,
                                                                               beta=beta,
                                                                               theta=theta_gt,
                                                                               seed=str(args.seed))
    root_name = 'objects/oup/' + str(task_name)
    if not os.path.exists(root_name):
        os.makedirs(root_name)

    prior = [Uniform(torch.zeros(1), torch.ones(1)),
             Uniform(-2 * torch.zeros(1), 2 * torch.ones(1))]
    simulator, prior = prepare_for_sbi(oup, prior)

    sum_net = OUPSummary(input_size=1, hidden_dim=64).to(device)
    neural_posterior = posterior_nn(
        model="maf",
        embedding_net=sum_net,
        hidden_features=20,
        num_transforms=3)

    inference = SNPE(prior=prior, density_estimator=neural_posterior, device=str(device))

    theta_gt = torch.tensor(theta_gt)
    obs = oup(theta_gt).to(device)
    sigma = torch.tensor(var)
    obs_cont = corruption.magnitude_sigma(obs, var=sigma, length=50).reshape(-1, 100, 50)

    theta, x = simulate_for_sbi(simulator, prior, num_simulations=num_simulations)
    x = x.reshape(num_simulations, 100, 50).to(device)
    theta = theta.to(device)
    density_estimator = inference.append_simulations(theta, x.unsqueeze(1)).train(
        corrupt_data_training=distance, x_obs=obs_cont)

    prior_new = [Uniform(-1 * torch.ones(1), 2 * torch.ones(1)),
                 Uniform(-4 * torch.zeros(1), 4 * torch.ones(1))]
    simulator, prior_new = prepare_for_sbi(oup, prior_new)
    posterior = inference.build_posterior(density_estimator, prior=prior_new)

    with open(root_name + "/posterior.pkl", "wb") as handle:
        pickle.dump(posterior, handle)

    torch.save(sum_net, root_name + "/sum_net.pkl")
    torch.save(density_estimator, root_name + "/density_estimator.pkl")

    with open(root_name + "/inference.pkl", "wb") as handle:
        pickle.dump(inference, handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--distance", type=str, default="mmd")
    parser.add_argument("--num_simulations", type=int, default=4000)
    parser.add_argument("--var", type=int, default=20)
    parser.add_argument("--theta", type=list, default=[0.5, 1.0])
    args = parser.parse_args()
    main(args)
