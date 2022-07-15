import argparse
import json
import os
import re
import copy
import numpy as np
from tqdm import tqdm
from lightsim2grid import LightSimBackend
from multiprocessing import Pool, Manager
import torch
import warnings

import grid2op
from grid2op.Agent import BaseAgent
from grid2op.utils import ScoreL2RPN2022
from grid2op.gym_compat import BoxGymActSpace, BoxGymObsSpace
    
from l2rpn_baselines.PPO_SB3.utils import SB3Agent

from GymEnvWithRecoWithDNWithShuffle import GymEnvWithRecoWithDNWithShuffle

DEFAULT_SAFE_MAX_RHO = 0.9
DEFAULT_TRAINING_ITER = 10_000_000
DEFAULT_LIMIT_CS_MARGINS = 150.

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


def cli():
    parser = argparse.ArgumentParser(description="Train baseline PPO")
    parser.add_argument("--has_cuda", default=1, type=int,
                        help="Is pytorch installed with cuda support ? (default True)")
    
    parser.add_argument("--cuda_device", default=0, type=int,
                        help="Which cuda device to use for pytorch (only used if you want to use cuda)")
    
    parser.add_argument("--safe_max_rho",
                        nargs='+',
                        help=(f"safe_max_rho to use for evaluation (default {DEFAULT_SAFE_MAX_RHO}). "
                              "You can add more than one."))
    
    parser.add_argument("--training_iter",
                        nargs='+',
                        help=(f"At which training iteration you want to load the agents ? (default "
                              f"{DEFAULT_TRAINING_ITER}). You can add more than one."))
    
    parser.add_argument("--limit_cs_margin", 
                        nargs='+',
                        help=(f"The margin used in `action.limit_curtail_storage(..., margin=XXX)` "
                              f"in the agent (default {DEFAULT_LIMIT_CS_MARGINS}). You can add more than one."))
    
    parser.add_argument("--agent_name", 
                        nargs='+',
                        help=f"Name for the agents you want to study, default to None, meaning 'i take everything'). You can add more than one.")
    
    parser.add_argument("--path_test_set",
                        default="../input_data_val",
                        type=str,
                        help="Name of the environment on which you want to evaluate your agents")
    
    parser.add_argument("--path_config_set",
                        default="../ingestion_program_val",
                        type=str,
                        help="Name of configuration file for the environment (mainly the seeds)")
    
    parser.add_argument("--path_agents", default="../../2022_ADPRL_paper/",
                        type=str,
                        help=("Path where the agents are stored. They are in the format `path_agents/A_DIRECTORY/AGENTS` "
                              "(ie in a sub directory of the path_agents argument)"))
    
    parser.add_argument("--expe_name", default="safe_max_rho",
                        type=str,
                        help=("Name to use for the experiment when saving the json data. Json will be saved in agents_runs_EXPE_NAME.json"))
    
    parser.add_argument("--nb_process", default=1, type=int,
                        help=("Number of processes used to run the experiment. Each agent is run on "
                              "a single process independantly from one another (default 1).")
                        )
    return parser.parse_args()


class BaselineAgent(BaseAgent):
    """see https://github.com/rte-france/l2rpn-baselines/issues/43
    a feature is missing in the l2rpn baselines repo"""
    def __init__(self, l2rpn_agent, limit_cs_margin):
        BaseAgent.__init__(self, l2rpn_agent.action_space)
        self.l2rpn_agent = l2rpn_agent
        self.limit_cs_margin = limit_cs_margin
    
    def act(self, obs, reward, done=False):
        action = self.l2rpn_agent.act(obs, reward, done)
        # We try to limit to end up with a "game over" because actions on curtailment or storage units.
        action.limit_curtail_storage(obs, margin=self.limit_cs_margin)
        return action


def get_agent(submission_dir, agent_dir, weights_dir, env, safe_max_rho, limit_cs_margin):
    """this is basically a copy paste of the PPO_SB3 evaluate function with some minor modification
    used to load the correct weights
    """
    
    # compute the score of said agent
    with open(os.path.join(submission_dir, "preprocess_obs.json"), 'r', encoding="utf-8") as f:
        obs_space_kwargs = json.load(f)
    with open(os.path.join(submission_dir, "preprocess_act.json"), 'r', encoding="utf-8") as f:
        act_space_kwargs = json.load(f)
    
    # load the attributes kept
    with open(os.path.join(agent_dir, "obs_attr_to_keep.json"), encoding="utf-8", mode="r") as f:
        obs_attr_to_keep = json.load(fp=f)
    with open(os.path.join(agent_dir, "act_attr_to_keep.json"), encoding="utf-8", mode="r") as f:
        act_attr_to_keep = json.load(fp=f)

    # create the action and observation space
    gym_observation_space =  BoxGymObsSpace(env.observation_space,
                                            attr_to_keep=obs_attr_to_keep,
                                            **obs_space_kwargs)
    gym_action_space = BoxGymActSpace(env.action_space,
                                      attr_to_keep=act_attr_to_keep,
                                      **act_space_kwargs)
    
    # create the gym environment for the PPO agent...
    gymenv = GymEnvWithRecoWithDNWithShuffle(env, safe_max_rho=float(safe_max_rho))    
    gymenv.action_space.close()
    gymenv.action_space = gym_action_space
    gymenv.observation_space.close()
    gymenv.observation_space = gym_observation_space
    
    # create a grid2gop agent based on that (this will reload the save weights)
    l2rpn_agent = SB3Agent(env.action_space,
                           gym_action_space,
                           gym_observation_space,
                           nn_path=weights_dir,
                           gymenv=gymenv
                           )
    
    agent_to_evaluate = BaselineAgent(l2rpn_agent, limit_cs_margin)
    return agent_to_evaluate


def create_env_score_fun(env_name, path_config_set):
    # create the environment
    env = grid2op.make(env_name,
                       backend=LightSimBackend())
    
    # read the seeds and other configuration
    config_file = os.path.join(os.path.abspath(path_config_set), "config_val.json")
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    env_seeds =  [int(config["episodes_info"][os.path.split(el)[-1]]["seed"]) for el in sorted(env.chronics_handler.real_data.subpaths)]
    
    # initialize the class to compute the losses
    score_fun = ScoreL2RPN2022(env,
                               env_seeds=env_seeds,
                               nb_scenario=int(config["nb_scenario"]),
                               min_losses_ratio=float(config["score_config"]["min_losses_ratio"]),
                               max_step=-1,
                               nb_process_stats=1)
    return env, score_fun


def get_agent_score(env_name,
                    submission_dir,
                    agent_dir,
                    weights_dir,
                    args,
                    safe_max_rho,
                    limit_cs_margin,
                    res,
                    safe_max_rho_str,
                    limit_cs_margin_str,
                    training_iter_str,
                    weights_dir_str,
                    count,
                    total):
    
    # create the env and the score function
    env, score_fun = create_env_score_fun(env_name, args.path_config_set)
    
    # create the agent
    agent_to_evaluate = get_agent(submission_dir,
                                  agent_dir,
                                  weights_dir,
                                  env,
                                  safe_max_rho,
                                  limit_cs_margin)
    
    # evaluate the agent
    scores, n_played, total_ts = score_fun.get(agent_to_evaluate,
                                               # path_save=path_save,  # TODO
                                               nb_process=1)
    # save the results
    this_run = {"score_avg": float(np.mean(scores)),
                "total_survived": float(np.sum(n_played)),
                "scores": [float(el) for el in scores], 
                "score_std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "n_played": [int(el) for el in n_played], 
                "total_ts": [int(el) for el in total_ts],
                }
    res[safe_max_rho_str][limit_cs_margin_str][training_iter_str][weights_dir_str] = this_run
    
    if count % total == 0:
        # save temporary results from time to time
        dictproxy_cls = type(res)
        res_json_serializable = convert_to_dict(res, dictproxy_cls)
        with open(f"agents_runs_{args.expe_name}_tmp_{count // total}.json", "w", encoding="utf-8") as f:
            json.dump(obj=res_json_serializable, fp=f)
        
    return this_run


def convert_to_dict(res, dictproxy_cls):
    json_serializable = {}
    for k in res.keys():
        res_k = res[k]
        if isinstance(res_k, dictproxy_cls):
            # multiprocessing object I cast to regular dict
            tmp = convert_to_dict(res_k, dictproxy_cls)
        else:
            # already a regular dict
            tmp = res_k
        json_serializable[str(k)] = tmp
        
    return json_serializable


def get_all_args(manager, safe_max_rhos, limit_cs_margins, training_iters, args):
    all_args = []
    res = manager.dict()
    i = 0
    for safe_max_rho_ in safe_max_rhos:    
        safe_max_rho = float(safe_max_rho_)
        safe_max_rho_str = safe_max_rho_
        res[safe_max_rho_str] = manager.dict()
        
        for limit_cs_margin_ in limit_cs_margins:
            limit_cs_margin = float(limit_cs_margin_)
            limit_cs_margin_str = limit_cs_margin_
            res[safe_max_rho_str][limit_cs_margin_str] = manager.dict()
                        
            for training_iter_ in training_iters:
                training_iter = int(training_iter_)
                training_iter_str = training_iter_
                res[safe_max_rho_str][limit_cs_margin_str][training_iter_str] = manager.dict()
            
                # loop for all agents
                root_dir = os.path.abspath(args.path_agents)
                for machine_dir in sorted(os.listdir(root_dir)):
                    submission_dir  = os.path.join(root_dir, machine_dir)
                    if not os.path.isdir(submission_dir):
                        # it is a regular file, we don't try to use it
                        continue
                    
                    for name in sorted(os.listdir(submission_dir)):
                        if tested_agent_names is not None:
                            # I this case I search if the possible agent name matches the one the user wants to keep
                            is_in = False
                            for el in tested_agent_names:
                                if re.search(f"{el}", name) is not None:
                                    # ignore the agents that do not have the right name
                                    is_in = True
                                    break
                                
                            if not is_in:
                                # I skip this name: the user does not want it
                                continue
                        
                        agent_dir = os.path.join(submission_dir, name)
                        weights_dir = os.path.join(agent_dir, f"{name}_{training_iter}_steps.zip")
                        if os.path.exists(weights_dir):
                            weights_dir_str = f"{weights_dir}"
                            all_args.append((str(args.path_test_set),
                                             submission_dir,
                                             agent_dir,
                                             weights_dir,
                                             args,
                                             safe_max_rho,
                                             limit_cs_margin,
                                             res,
                                             safe_max_rho_str,
                                             limit_cs_margin_str,
                                             training_iter_str,
                                             weights_dir_str,
                                             i,
                                             int(args.nb_process)
                                             )
                                            )
                            i += 1
    return res, all_args
                            
if __name__ == "__main__":
    args = cli()
    use_cuda = check_cuda(args)
    torch.multiprocessing.set_start_method("spawn")
    
    # create the "manager" that will hold the data
    manager = Manager()
    
    # read the arguments for the experiment you want to run
    if args.safe_max_rho is None:
        safe_max_rhos = [DEFAULT_SAFE_MAX_RHO]
    else:
        safe_max_rhos = copy.deepcopy(args.safe_max_rho)
    
    if args.training_iter is None:
        training_iters = [DEFAULT_TRAINING_ITER]
    else:
        training_iters = copy.deepcopy(args.training_iter)
    
    if args.limit_cs_margin is None:
        limit_cs_margins = [DEFAULT_LIMIT_CS_MARGINS]
    else:
        limit_cs_margins = copy.deepcopy(args.limit_cs_margin)
    
    tested_agent_names = copy.deepcopy(args.agent_name)
    
    # compute the arguments needed for the function
    res, all_args = get_all_args(manager, safe_max_rhos, limit_cs_margins, training_iters, args)
    
    # execute the agents
    print()
    print(f"Executing {len(all_args)} agents on {args.nb_process} process")
    with Pool(int(args.nb_process)) as p:
        p.starmap(get_agent_score, all_args)
    
    # I save the score of each agents
    dictproxy_cls = type(res)
    res_json_serializable = convert_to_dict(res, dictproxy_cls)
    with open(f"agents_runs_{args.expe_name}.json", "w", encoding="utf-8") as f:
        json.dump(obj=res_json_serializable, fp=f)
