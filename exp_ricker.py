from simulators.ricker import ricker

from networks.summary_nets import RickerSummary
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
    corrupt = args.corruption
    degree = args.degree
    distance = args.distance
    beta = args.beta
    task_name = "{corruption}_degree={degree}_var=15_{distance}_beta={beta}_theta38".format(corruption=corrupt,
                                                                               degree=degree,
                                                                               distance=distance,
                                                                               beta=beta,
                                                                               )
    root_name = 'objects/' + str(task_name)
    if not os.path.exists(root_name):
        os.mkdir(root_name)
    root_name_with_seed = root_name + "/{seed}".format(seed=int(args.seed))
    if not os.path.exists(root_name_with_seed):
        os.mkdir(root_name_with_seed)

    low = torch.tensor([2., 0.]).to(device)
    high = torch.tensor([8., 20.]).to(device)
    prior = torch.distributions.uniform.Uniform(low, high)

    num_simulations = 4000
    # theta, x = simulate_for_sbi(ricker, prior, num_simulations=num_simulations)
    # np.save("x_10.npy", x.cpu().detach().numpy())
    # np.save("theta_10.npy", theta.cpu().detach().numpy())
    # x = torch.tensor(x).reshape(num_simulations, 100, 100).to(device)
    # theta = torch.tensor(theta).to(device)

    x = torch.tensor(np.load("data/x.npy")).reshape(num_simulations, 100, 100).to(device)
    x = torch.tensor(np.load("data/x_corrupted.npy")).to(device)

    theta = torch.tensor(np.load("data/theta.npy")).to(device)

    # normal model
    sum_net_normal = RickerSummary(input_size=1, hidden_dim=4).to(device)
    neural_posterior_normal = posterior_nn(
        model="maf",
        embedding_net=sum_net_normal,
        hidden_features=20,
        num_transforms=3)

    inference_normal = SNPE(prior=prior, density_estimator=neural_posterior_normal, device=str(device))
    density_estimator_normal = inference_normal.append_simulations(theta, x).train(corrupt_data_training="none")
    posterior_normal = inference_normal.build_posterior(density_estimator_normal)

    torch.save(sum_net_normal, root_name_with_seed + "/sum_net_normal.pkl")
    torch.save(density_estimator_normal, root_name_with_seed + "/density_estimator_normal.pkl")
    # with open(root_name + "/inference_normal.pkl", "wb") as handle:
    #     pickle.dump(inference_normal, handle)
    with open(root_name_with_seed + "/posterior_normal.pkl", "wb") as handle:
        pickle.dump(posterior_normal, handle)

    # robust
    sum_net_robust = RickerSummary(input_size=1, hidden_dim=4).to(device)
    neural_posterior_robust = posterior_nn(
        model="maf",
        embedding_net=sum_net_robust,
        hidden_features=20,
        num_transforms=3)
    theta = theta.to(device)
    x = x.to(device)
    inference_robust = SNPE(prior=prior, density_estimator=neural_posterior_robust, device=str(device))
    density_estimator_robust = inference_robust.append_simulations(theta, x).train(corrupt_data_training=distance,
                                                                                   corruption_method=corrupt,
                                                                                   beta=beta,
                                                                                   corruption_degree=degree)
    posterior_robust = inference_robust.build_posterior(density_estimator_robust)

    torch.save(sum_net_robust, root_name_with_seed+"/sum_net_robust.pkl")
    torch.save(density_estimator_robust, root_name_with_seed+"/density_estimator_robust.pkl")
    # with open(root_name+"/inference_robust.pkl", "wb") as handle:
    #     pickle.dump(inference_robust, handle)
    with open(root_name_with_seed+"/posterior_robust.pkl", "wb") as handle:
        pickle.dump(posterior_robust, handle)


def new_exp(args):
    corrupt = args.corruption
    distance = args.distance
    beta = args.beta

    task_name = "{corruption}_degree={degree}_var=15_{distance}_beta={beta}_theta38".format(corruption=corrupt,
                                                                               distance=distance,
                                                                               beta=beta,
                                                                               )
    root_name = 'objects/' + str(task_name)
    if not os.path.exists(root_name):
        os.mkdir(root_name)
    root_name_with_seed = root_name + "/{seed}".format(seed=int(args.seed))
    if not os.path.exists(root_name_with_seed):
        os.mkdir(root_name_with_seed)

    num_rounds = 1
    num_simulations = 1000
    prior = [Uniform(2 * torch.ones(1), 8 * torch.ones(1)),
             Uniform(torch.zeros(1), 20 * torch.ones(1))]
    simulator, prior = prepare_for_sbi(ricker, prior)

    sum_net_normal = RickerSummary(input_size=1, hidden_dim=4).to(device)
    neural_posterior_normal = posterior_nn(
        model="maf",
        embedding_net=sum_net_normal,
        hidden_features=20,
        num_transforms=3)

    inference_normal = SNPE(prior=prior, density_estimator=neural_posterior_normal, device=str(device))
    posteriors = []
    theta_gt = torch.tensor([3, 8])
    x_o = ricker(theta_gt).to(device)
    sigma = torch.tensor([80])
    x_o_cont = corruption.magnitude_sigma(x_o, var=sigma, length=100).reshape(-1, 100, 100)
    # x_o_cont = torch.tensor(np.load("data/x_o_cont.npy"))
    proposal = prior
    for i in range(num_rounds):
        theta, x = simulate_for_sbi(simulator, proposal, num_simulations=num_simulations)
        x = torch.tensor(x).reshape(num_simulations, 100, 100).to(device)
        theta = torch.tensor(theta).to(device)
        density_estimator_normal = inference_normal.append_simulations(theta, x.unsqueeze(1), proposal=proposal).train(corrupt_data_training="mmd", x_obs=x_o_cont)

        prior_new = [Uniform(2 * torch.ones(1), 8 * torch.ones(1)),
                     Uniform(torch.zeros(1), 50 * torch.ones(1))]
        simulator, prior_new = prepare_for_sbi(ricker, prior_new)
        posterior_normal = inference_normal.build_posterior(density_estimator_normal, prior=prior_new)
        # posteriors.append(posterior_normal)
        # proposal = posterior_normal.set_default_x(x_o_cont.reshape(1, 1, 100, 100))

        with open(root_name_with_seed + "/posterior_normal_{r}.pkl".format(r=str(i)), "wb") as handle:
            pickle.dump(posterior_normal, handle)

        torch.save(sum_net_normal, root_name_with_seed + "/sum_net_normal_{r}.pkl".format(r=str(i)))
        torch.save(density_estimator_normal, root_name_with_seed + "/density_estimator_normal_{r}.pkl".format(r=str(i)))

        # with open(root_name_with_seed + "/inference_normal_{r}.pkl".format(r=str(i)), "wb") as handle:
        #     pickle.dump(inference_normal, handle)
    # with open(root_name_with_seed + "/posterior_robust.pkl", "wb") as handle:
    #     pickle.dump(posterior_normal, handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corruption", type=str, default="magnitude_Gaussian")
    parser.add_argument("--beta", type=float, default=1)
    # parser.add_argument("--distance", type=str, default="pre_generated_sigma")
    parser.add_argument("--seed", type=int, default="0")
    parser.add_argument("--distance", type=str, default="obs_minimize")
    parser.add_argument("--simulator", type=str, default="ricker")

    args = parser.parse_args()
    new_exp(args)
