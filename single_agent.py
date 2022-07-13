from grid2op.Agent import BaseAgent
from l2rpn_baselines.PPO_SB3 import evaluate
import os
import json
from GymEnvWithRecoWithDNWithShuffle import GymEnvWithRecoWithDNWithShuffle

class BaselineAgent(BaseAgent):
  def __init__(self, l2rpn_agent):
    self.l2rpn_agent = l2rpn_agent
    BaseAgent.__init__(self, l2rpn_agent.action_space)
  
  def act(self, obs, reward, done=False):
    action = self.l2rpn_agent.act(obs, reward, done)
    # We try to limit to end up with a "game over" because actions on curtailment or storage units.
    action.limit_curtail_storage(obs, margin=100)
    return action


def make_agent(env, submission_dir, agent_name):
    """
    This function will be used by codalab to create your agent. It should accept exactly an environment and a path
    to your submission directory and return a valid agent.
    """

    agent_dir = os.path.join(submission_dir, "saved_model")
    normalization_dir = os.path.join(submission_dir, "normalization")

    with open(os.path.join(normalization_dir, "preprocess_obs.json"), 'r', encoding="utf-8") as f:
      obs_space_kwargs = json.load(f)
    with open(os.path.join(normalization_dir, "preprocess_act.json"), 'r', encoding="utf-8") as f:
      act_space_kwargs = json.load(f)

    l2rpn_agent, _ = evaluate(env,
                    nb_episode=0,
                    load_path=agent_dir,
                    name=agent_name,
                    gymenv_class=GymEnvWithRecoWithDNWithShuffle,
                    gymenv_kwargs={"safe_max_rho": 0.95},
                    obs_space_kwargs=obs_space_kwargs,
                    act_space_kwargs=act_space_kwargs)

    return BaselineAgent(l2rpn_agent)


if __name__ == "__main__":
  from lightsim2grid import LightSimBackend
  import grid2op

  env = grid2op.make("input_data_local", backend=LightSimBackend())
  agent_set = make_agent(env, ".", "PPO_agent1_20220709_152030")

  nb_steps = 0
  obs = env.reset()
  done = False
  reward = 0
  while not done:
    nb_steps += 1
    action = agent_set.act(obs, reward, done)
    print(action)
    obs, reward, done, _ = env.step(action)
  
  print(f"Nb steps: {nb_steps}")