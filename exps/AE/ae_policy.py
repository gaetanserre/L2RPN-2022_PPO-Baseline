import torch as th
import numpy as np

from gym.spaces.box import Box

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import Distribution
from typing import Tuple

class AEMlpPolicy(ActorCriticPolicy):
  def __init__(
      self,
      observation_space,
      action_space,
      lr_schedule,
      ae_weights,
      latent_dim,
      **kwargs
  ):

    # Modify the observation space to be the latent space
    observation_space = Box(0, 1, (latent_dim,), np.float32)
    super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    # Load autoencoder and disable its training
    self.ae = th.jit.load(ae_weights).encoder
    self.ae.eval()
    for param in self.ae.parameters():
      param.requires_grad = False
  
  def forward(self, obs: th.Tensor, **kwargs) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    h = self.ae(obs)
    return super().forward(h, **kwargs)

  def _predict(self, observation: th.Tensor, **kwargs) -> th.Tensor:
    h = self.ae(observation)
    return super()._predict(h, **kwargs)
  
  def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    h = self.ae(obs)
    return super().evaluate_actions(h, actions)
  
  def get_distribution(self, obs: th.Tensor) -> Distribution:
    h = self.ae(obs)
    return super().get_distribution(h)
  
  def predict_values(self, obs: th.Tensor) -> th.Tensor:
    h = self.ae(obs)
    return super().predict_values(h)

  def get_parameters(self, trainable=False):
    return sum(p.numel() for p in self.parameters() if not trainable or p.requires_grad)
