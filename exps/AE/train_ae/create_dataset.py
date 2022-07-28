import grid2op
from lightsim2grid import LightSimBackend
import numpy as np
import argparse
from tqdm import tqdm
from grid2op.Agent import RecoPowerlineAgent
from l2rpn_baselines.utils import GymEnvWithRecoWithDN
from grid2op.gym_compat import BoxGymObsSpace
import json
import pandas as pd
import os

def balance_dataset(observations, flags):
  dataset = {"observations": observations, "flags": flags}
  df = pd.DataFrame.from_dict(dataset)

  flag_values = ["safe", "medium", "danger"]
  nb_obs_per_flag = np.array([(df["flags"] == flag).sum() for flag in flag_values])
  nb_min = nb_obs_per_flag.min()
  idx_flag_min = nb_obs_per_flag.argmin()


  for i, flag in enumerate(flag_values):
    if i == idx_flag_min: continue

    index = df[df["flags"]==flag].index
    nb_to_remove = len(index) - nb_min
    df.drop(np.random.choice(index, nb_to_remove, replace=False), inplace=True)
  return df["observations"].to_list(), df["flags"].to_list()

def cli():
  parser = argparse.ArgumentParser(description="Create observations dataset")
  parser.add_argument("--env_name", required=True, type=str,
                      help="The name of the environment from which to select the scenarios")
  parser.add_argument("--n_obs", required=True, type=int,
                      help="The number of observations to save in the dataset")
  parser.add_argument("--balance", default=1, type=int,
                      help="Whether to balance the data set or not")
  
  return parser.parse_args()


if __name__ == "__main__":
  args = cli()

  env = grid2op.make(args.env_name, backend=LightSimBackend())
  env.chronics_handler.real_data.shuffle()
  
  with open(os.path.join(".", "normalization", "preprocess_obs.json"), "r", encoding="utf-8") as f:
    obs_space_kwargs = json.load(f)
  obs_attr_to_keep = ["month", "day_of_week", "hour_of_day", "minute_of_hour",
                                  "gen_p", "load_p", 
                                  "p_or", "rho", "timestep_overflow", "line_status",
                                  # dispatch part of the observation
                                  "actual_dispatch", "target_dispatch",
                                  # storage part of the observation
                                  "storage_charge", "storage_power",
                                  # curtailment part of the observation
                                  "curtailment", "curtailment_limit",  "gen_p_before_curtail",
                                  ]
  obs_to_gym = BoxGymObsSpace(env.observation_space,
                              attr_to_keep=obs_attr_to_keep,
                              **obs_space_kwargs)


  agent = RecoPowerlineAgent(env.action_space)

  n_obs = 0
  observations = []
  flags        = []
  with tqdm(total=args.n_obs) as pbar:
    while n_obs < args.n_obs:
      obs = env.reset()
      done = False
      while not done:
        n_obs += 1
        observations.append(obs_to_gym.to_gym(obs))
        if obs.rho.max() < 0.70: flags.append("safe")
        elif obs.rho.max() > 0.90: flags.append("danger")
        else: flags.append("medium")

        if n_obs >= args.n_obs: break

        act = agent.act(obs, None, None)
        obs, reward, done, info = env.step(act)
        pbar.update()

  if args.balance:
    observations, flags = balance_dataset(observations, flags)

  np.savez_compressed(
        f"./dataset_{len(observations)}obs",
        observations=observations,
        flags=flags
    )
  