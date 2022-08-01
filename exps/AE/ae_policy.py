import re
import torch as th
import numpy as np
import copy

from gym.spaces.box import Box

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import Distribution
from typing import Dict, Optional, Tuple, Union
from stable_baselines3.common.preprocessing import is_image_space, maybe_transpose
from stable_baselines3.common.utils import obs_as_tensor

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

    self.latent_dim = latent_dim
    # Modify the observation space to be the latent space
    observation_space = Box(0, 1, (self.latent_dim,), np.float32)
    super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    # Load autoencoder and disable its training
    self.ae = th.jit.load(ae_weights).encoder
    self.ae.eval()
    for param in self.ae.parameters():
      param.requires_grad = False
  
  def _encode(self, obs):
    need_encode = False
    if len(obs.shape) == 1:
      need_encode = obs.shape[0] > self.latent_dim
    else:
      need_encode = obs.shape[1] > self.latent_dim
    
    if need_encode:
      return self.ae(obs)
    else:
      return obs


  def forward(self, obs: th.Tensor, **kwargs) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    h = self._encode(obs)
    return super().forward(h, **kwargs)


  def _predict(self, observation: th.Tensor, **kwargs) -> th.Tensor:
    h = self._encode(observation)
    return super()._predict(h, **kwargs)
  

  def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    h = self._encode(obs)
    return super().evaluate_actions(h, actions)
  

  def get_distribution(self, obs: th.Tensor) -> Distribution:
    h = self._encode(obs)
    return super().get_distribution(h)
  

  def predict_values(self, obs: th.Tensor) -> th.Tensor:
    h = self._encode(obs)
    return super().predict_values(h)


  def obs_to_tensor(self, observation: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tuple[th.Tensor, bool]:
        """
        Convert an input observation to a PyTorch tensor that can be fed to a model.
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :return: The observation as PyTorch tensor
            and whether the observation is vectorized or not
        """
        vectorized_env = len(observation.shape) > 1
        if isinstance(observation, dict):
            # need to copy the dict as the dict in VecFrameStack will become a torch tensor
            observation = copy.deepcopy(observation)
            for key, obs in observation.items():
                obs_space = self.observation_space.spaces[key]
                if is_image_space(obs_space):
                    obs_ = maybe_transpose(obs, obs_space)
                else:
                    obs_ = np.array(obs)
                # Add batch dimension if needed
                observation[key] = obs_.reshape((-1,) + self.observation_space[key].shape)

        elif is_image_space(self.observation_space):
            # Handle the different cases for images
            # as PyTorch use channel first format
            observation = maybe_transpose(observation, self.observation_space)

        else:
            observation = np.array(observation)

        if not isinstance(observation, dict):
            # Dict obs need to be handled separately
            # Add batch dimension if needed
            observation = observation.reshape((-1,) + observation.shape)

        observation = th.tensor(observation).to(self.device)
        return observation, vectorized_env


  def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

    return super().predict(observation, state, episode_start, deterministic)


  def get_parameters(self, trainable=False):
    return sum(p.numel() for p in self.parameters() if not trainable or p.requires_grad)
