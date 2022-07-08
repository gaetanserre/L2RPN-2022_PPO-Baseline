import grid2op
from l2rpn_baselines.PPO_SB3 import train
from grid2op.Reward import EpisodeDurationReward
from lightsim2grid import LightSimBackend
from grid2op.Chronics import MultifolderWithCache
from GymEnvWithRecoWithDNWithShuffle import GymEnvWithRecoWithDNWithShuffle
from grid2op.Parameters import Parameters
import torch
import datetime
import json
import os
import argparse
import warnings
import numpy as np


def cli():
    parser = argparse.ArgumentParser(description="Train baseline PPO")
    parser.add_argument("--has_cuda", default=1, type=int,
                        help="Is pytorch installed with cuda support ? (default True)")
    
    parser.add_argument("--cuda_device", default=0, type=int,
                        help="Which cuda device to use for pytorch (only used if you want to use cuda)")

    parser.add_argument("--nb_training", default=-1, type=int,
                        help="How many models do you want to train ? (default: -1 for infinity, might take a while...)")

    parser.add_argument("--lr", default=3e-6, type=float,
                        help="learning rate to use (default 3e-6)")
    
    parser.add_argument("--safe_max_rho", default=0.2, type=float,
                        help="safe_max_rho to use for training (default 0.2)")
    
    parser.add_argument("--training_iter", default=10_000_000, type=int,
                        help="Number of training 'iteration' to perform (default 10_000_000)")
    
    parser.add_argument("--agent_name", default="PPO_agent", type=str,
                        help="Name for your agent, default 'PPO_agent'")
    
    parser.add_argument("--seed", default=-1, type=int,
                        help="Seed to use (default to -1 meaning 'don't seed the env') for the environment (same seed used to train all agents)")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = cli()
    use_cuda = int(args.has_cuda) >= 1
    if use_cuda >= 1:
        assert torch.cuda.is_available(), "cuda is not available on your machine with pytorch"
        torch.cuda.set_device(int(args.cuda_device))
    else:
        warnings.warn("You won't use cuda")
        if int(args.cuda_device) != 0:
            warnings.warn("You specified to use a cuda_device (\"--cuda_device = XXX\") yet you tell the program not to use cuda (\"--has_cuda = 0\"). "
                          "This program will ignore the \"--cuda_device = XXX\" directive.")

    # we use a dictionary to set the training parameters of the agent
    train_args = {}
    train_args["logs_dir"] = "./logs"
    train_args["save_path"] = "agents"
    train_args["verbose"] = 1

    # this class contains the two expert rules described in the paper
    train_args["gymenv_class"] = GymEnvWithRecoWithDNWithShuffle

    train_args["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # hyperparameters
    train_args["obs_attr_to_keep"] = ["month", "day_of_week", "hour_of_day", "minute_of_hour",
                                      "gen_p", "load_p", 
                                      "p_or", "rho", "timestep_overflow", "line_status",
                                      # dispatch part of the observation
                                      "actual_dispatch", "target_dispatch",
                                      # storage part of the observation
                                      "storage_charge", "storage_power",
                                      # curtailment part of the observation
                                      "curtailment", "curtailment_limit",  "gen_p_before_curtail",
                                      ]
    train_args["act_attr_to_keep"] = ["curtail", "set_storage"]
    train_args["iterations"] = int(args.training_iter)
    train_args["learning_rate"] = float(args.lr)
    train_args["net_arch"] = [300, 300, 300]
    train_args["gamma"] = 0.999
    train_args["gymenv_kwargs"] = {"safe_max_rho": float(args.safe_max_rho)}
    train_args["normalize_act"] = True
    train_args["normalize_obs"] = True

    train_args["save_every_xxx_steps"] = min(train_args["iterations"] // 10, 500_000)

    train_args["n_steps"] = 16
    train_args["batch_size"] = 16

    # limit the curtailment and storage actions. Same as action.limit_curtail_storage(obs, margin=m)
    param = Parameters()
    param.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION = True

    # determine how many agent will be trained
    nb_train = args.nb_training
    if nb_train == -1:
        nb_train = int(np.iinfo(np.int64).max)

    # define the reward
    reward_class = EpisodeDurationReward

    # create train environment
    env_train = grid2op.make("l2rpn_wcci_2022",
                          reward_class=reward_class,
                          backend=LightSimBackend(),
                          chronics_class=MultifolderWithCache,
                          param=param)

    env_train.chronics_handler.real_data.set_filter(lambda x: True)
    # do not forget to load all the data in memory !
    env_train.chronics_handler.real_data.reset()



    # Now do the loop to train the agents
    for _ in range(nb_train):
        if int(args.seed) >= 0:
            env_train.seed(args.seed)
        # reset the env to have everything "correct"
        env_train.reset()
        # shuffle the order of the chronics
        env_train.chronics_handler.shuffle()
        # reset the env to have everything "correct"
        env_train.reset()
        
        # assign a unique name
        agent_name = f"{args.agent_name}_{datetime.datetime.now():%Y-%m-%d_%H-%M}"
        train_args["name"] = agent_name

        # Loading data for normalization of observations and actions
        with open(os.path.join("normalization", "preprocess_obs.json"), "r", encoding="utf-8") as f:
            obs_space_kwargs = json.load(f)
        with open(os.path.join("normalization", "preprocess_act.json"), "r", encoding="utf-8") as f:
            act_space_kwargs = json.load(f)

        ppo_agent = train(env_train,
                  obs_space_kwargs=obs_space_kwargs,
                  act_space_kwargs=act_space_kwargs,
                  **train_args)