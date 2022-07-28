from l2rpn_baselines.PPO_SB3 import train
from stable_baselines3.ppo import MlpPolicy
import torch as th
from typing import Tuple
from gym.spaces.box import Box
import numpy as np

class AEMlpPolicy(MlpPolicy):
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

  def get_parameters(self, trainable=False):
    return sum(p.numel() for p in self.parameters() if not trainable or p.requires_grad)
