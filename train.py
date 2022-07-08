import grid2op
from l2rpn_baselines.PPO_SB3 import train
from grid2op.Reward import EpisodeDurationReward
from lightsim2grid import LightSimBackend
from grid2op.Chronics import MultifolderWithCache
from l2rpn_baselines.utils import GymEnvWithRecoWithDN
from grid2op.Parameters import Parameters
import torch
import datetime
import json
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training program")
    parser.add_argument("-d", "--device", required=False,
                        default=0, type=int,
                        help="GPU to use for training")
    args = parser.parse_args()


    # We use a dictionary to set the training parameters of the agent
    train_args = {}
    train_args["logs_dir"] = "./logs"
    train_args["save_path"] = "agents"
    train_args["name"] = '_'.join(["PPO_agent", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')])
    train_args["verbose"] = 1

    # This class contains the two expert rules described in the paper
    train_args["gymenv_class"] = GymEnvWithRecoWithDN

    train_args["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
      torch.cuda.set_device(args.device)

    # Hyperparameters
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
    train_args["iterations"] = 10_000_000
    train_args["learning_rate"] = 3e-6
    train_args["net_arch"] = [300, 300, 300]
    train_args["gamma"] = 0.999
    train_args["gymenv_kwargs"] = {"safe_max_rho": 0.2}
    train_args["normalize_act"] = True
    train_args["normalize_obs"] = True

    train_args["save_every_xxx_steps"] = min(train_args["iterations"] // 10, 500_000)

    train_args["n_steps"] = 16
    train_args["batch_size"] = 16

    # Limit the curtailment and storage actions. Same as action.limit_curtail_storage(obs, margin=m)
    p = Parameters()
    p.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION = True


    # Define the reward
    reward_class = EpisodeDurationReward

    # Create train environment
    env_train = grid2op.make("l2rpn_wcci_2022",
                      reward_class=reward_class,
                      backend=LightSimBackend(),
                      chronics_class=MultifolderWithCache,
                      param=p)
    env_train.chronics_handler.real_data.reset()

    # Loading data for normalization of observations and actions
    with open(os.path.join("normalization", "preprocess_obs.json"), "r", encoding="utf-8") as f:
        obs_space_kwargs = json.load(f)
    with open(os.path.join("normalization", "preprocess_act.json"), "r", encoding="utf-8") as f:
        act_space_kwargs = json.load(f)

    # Train the agent
    ppo_agent = train(env_train,
                  obs_space_kwargs=obs_space_kwargs,
                  act_space_kwargs=act_space_kwargs,
                  **train_args)