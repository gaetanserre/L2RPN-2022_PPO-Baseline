from grid2op.Agent import BaseAgent
import numpy as np
import json
import os
from l2rpn_baselines.PPO_SB3 import evaluate
from .GymEnvWithRecoWithDNWithShuffle import GymEnvWithRecoWithDNWithShuffle
from .single_agent import BaselineAgent

class MixtureExperts(BaseAgent):
  def __init__(self, agents):
    assert len(agents) > 0
    self.agents = agents
    BaseAgent.__init__(self, self.agents[0].l2rpn_agent.action_space)
  
  def act(self, obs, reward, done=False):
    actions = list(map(lambda agent: agent.act(obs, reward, done), self.agents))
    sim_rewards = np.array(map(lambda act: obs.simulate(act, 0)[1], actions))
    return actions[np.argmax(sim_rewards)]


def make_agent(env, submission_dir, agents_dir):
    """
    This function will be used by codalab to create your agent. It should accept exactly an environment and a path
    to your submission directory and return a valid agent.
    """

    normalization_dir = os.path.join(submission_dir, "normalization")
    agents_dir        = os.path.join(submission_dir, agents_dir)

    with open(os.path.join(normalization_dir, "preprocess_obs.json"), 'r', encoding="utf-8") as f:
      obs_space_kwargs = json.load(f)
    with open(os.path.join(normalization_dir, "preprocess_act.json"), 'r', encoding="utf-8") as f:
      act_space_kwargs = json.load(f)


    agents = []
    for d in os.listdir(agents_dir):
      if os.path.isdir(os.path.join(agents_dir, d)):
        l2rpn_agent, _ = evaluate(env,
                      nb_episode=0,
                      load_path=agents_dir,
                      name=d,
                      gymenv_class=GymEnvWithRecoWithDNWithShuffle,
                      gymenv_kwargs={"safe_max_rho": 1.0},
                      obs_space_kwargs=obs_space_kwargs,
                      act_space_kwargs=act_space_kwargs)
        agents.append(BaselineAgent(l2rpn_agent))
  
    return MixtureExperts(agents)


if __name__ == "__main__":
  from lightsim2grid import LightSimBackend
  import grid2op

  env = grid2op.make("input_data_local", backend=LightSimBackend())
  agent_set = make_agent(env, ".", "saved_model")

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