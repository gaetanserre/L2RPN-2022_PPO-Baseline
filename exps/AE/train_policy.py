import pdb
import warnings
import torch
import datetime
import sys
import re
import os
import copy
import argparse
import numpy as np

from lightsim2grid import LightSimBackend

import grid2op
from grid2op.Chronics import MultifolderWithCache
from grid2op.utils import ScoreL2RPN2022

from src.utils import *

from grid2op.Reward import EpisodeDurationReward

from src.GymEnvWithRecoWithDNWithShuffle import GymEnvWithRecoWithDNWithShuffle

from ae_policy import AEMlpPolicy


ENV_NAME = "l2rpn_wcci_2022"


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
    
    parser.add_argument("--ratio_keep_chronics", default=1., type=float,
                        help=("Faction of the training set to keep for training. Chronics will be re sampled for each agents. Note that each agent "
                              "will see different chronics (sampling is done for each agent)")
                        )
    
    parser.add_argument("--chronics_name",
                        nargs='+',
                        help="Chronics to use for training")
    
    return parser.parse_args()


def check_cuda(args):
    use_cuda = int(args.has_cuda) >= 1
    
    if use_cuda >= 1:
        assert torch.cuda.is_available(), "cuda is not available on your machine with pytorch"
        torch.cuda.set_device(int(args.cuda_device))
    else:
        warnings.warn("You won't use cuda")
        if int(args.cuda_device) != 0:
            warnings.warn("You specified to use a cuda_device (\"--cuda_device = XXX\") yet you tell the program not to use cuda (\"--has_cuda = 0\"). "
                          "This program will ignore the \"--cuda_device = XXX\" directive.")
    return use_cuda        
            
            
if __name__ == "__main__":
    args = cli()
    use_cuda = check_cuda(args)
    
    # Split sets and statistics parameters
    is_windows = sys.platform.startswith("win32")
    is_windows_or_darwin = sys.platform.startswith("win32") or sys.platform.startswith("darwin")
    nb_process_stats = 8 if not is_windows_or_darwin else 1
    deep_copy = is_windows  # force the deep copy on windows (due to permission issue in symlink in windows)
    verbose = 1
    SCOREUSED = ScoreL2RPN2022
    name_stats = "_reco_powerline"

    # save / load information (NB agent name is defined later)
    env_name_train = '_'.join([ENV_NAME, "train"])
    save_path = "./saved_model"
    gymenv_class = GymEnvWithRecoWithDNWithShuffle
    load_path = None
    load_name = None

    # PPO parameters
    train_args = {}
    train_args["logs_dir"] = "./logs"
    train_args["save_path"] = save_path
    train_args["verbose"] = 1
    train_args["gymenv_class"] = gymenv_class
    train_args["device"] = torch.device("cuda" if use_cuda else "cpu")
    # some "meta parameters" of the training and the optimization
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
    train_args["net_arch"] = [300, 300, 300]
    train_args["gamma"] = 0.999
    train_args["gymenv_kwargs"] = {"safe_max_rho": float(args.safe_max_rho)}
    train_args["normalize_act"] = True
    train_args["normalize_obs"] = True
    train_args["save_every_xxx_steps"] = min(max(train_args["iterations"]//10, 1), 1_000_000)
    train_args["n_steps"] = 16
    train_args["batch_size"] = 16
    train_args["learning_rate"] = float(args.lr)
    train_args["model_policy"] = AEMlpPolicy
    train_args["policy_kwargs"] = {"ae_weights": "train_ae/ae_weights.pty", "latent_dim": 400}
    
    # Set the right grid2op environment parameters
    filter_chronics = None        
    try:
        if filter_chronics is None:
            # env = grid2op.make(ENV_NAME)
            nm_train, nm_val, nm_test = split_train_val_test_sets(ENV_NAME, deep_copy)
            generate_statistics([nm_val, nm_test], SCOREUSED, nb_process_stats, name_stats, verbose)
        else:
            generate_statistics([ENV_NAME], SCOREUSED, nb_process_stats, name_stats, verbose, filter_fun=filter_chronics)
    except Exception as exc_:
        if str(exc_).startswith("Impossible to create"):
            pass
        else:
            raise exc_
    env_tmp = grid2op.make(env_name_train if filter_chronics is None else ENV_NAME)            
    param = env_tmp.parameters
    param.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION = True
    
    size_ = None
    if float(args.ratio_keep_chronics) < 1.:
        all_data = env_tmp.chronics_handler.real_data.subpaths
        nb_data = len(all_data)
        size_ = int(float(args.ratio_keep_chronics) * nb_data)
    
    # determine how many agent will be trained
    nb_train = args.nb_training
    if nb_train == -1:
        nb_train = int(np.iinfo(np.int64).max)
    
    # prepare the real training environment
    env_train = grid2op.make(env_name_train if filter_chronics is None else ENV_NAME,
                             reward_class=EpisodeDurationReward,
                             backend=LightSimBackend(),
                             chronics_class=MultifolderWithCache,
                             param=param)
    
    if args.chronics_name is not None:
        chron_to_keep = copy.deepcopy(args.chronics_name)
        def filter_chronics(x, li_to_keep=chron_to_keep):
            res = False
            for el in li_to_keep:
                if re.search(el, x) is not None:
                    res = True
                    break
            return res
        
    if filter_chronics is not None:
        env_train.chronics_handler.real_data.set_filter(filter_chronics)
    else:
        env_train.chronics_handler.real_data.set_filter(lambda x: True)
        
    # do not forget to load all the data in memory !
    if float(args.ratio_keep_chronics) >= 1. or args.chronics_name is not None:
        # otherwise it's reset for each agent
        env_train.chronics_handler.real_data.reset()

    # now do the loop to train the agents
    for _ in range(nb_train):
        if float(args.ratio_keep_chronics) < 1. and args.chronics_name is None:
            # TODO reproductibility !
            ID_TO_KEEP = set(np.random.choice(all_data, size=size_, replace=False))
            def filter_chronics(nm, to_keep=ID_TO_KEEP):
                return nm in to_keep
            
            env_train.chronics_handler.real_data.set_filter(filter_chronics)
            env_train.chronics_handler.real_data.reset()
            print(env_train.chronics_handler.real_data._order)
        
        if int(args.seed) >= 0:
            env_train.seed(args.seed)
        # reset the env to have everything "correct"
        env_train.reset()
        # shuffle the order of the chronics
        env_train.chronics_handler.shuffle()
        # reset the env to have everything "correct"
        env_train.reset()
        # assign a unique name
        agent_name = f"{args.agent_name}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
        train_args["name"] = agent_name

        agent = train_agent(env_train, train_args)