import dmlab2d
from gymnasium import spaces
import numpy as np
from ray.rllib.env import multi_agent_env

from baselines.train import utils

PLAYER_STR_FORMAT = 'player_{index}'

# generate alphas and betas

def rearrange_IA(rewards, alphas, betas, gamma, lmbda, cur_e):
  N = len(rewards)
  coeff = 1 / (N - 1)
  reward_vals = np.array(list(rewards.values()))

  next_e = reward_vals + gamma * lmbda * cur_e

  new_reward_vals = np.array([rewi - coeff * alpha * np.max(reward_vals - rewi)\
                    - coeff * beta * np.max(rewi - reward_vals) 
                    for rewi, alpha, beta in zip(list(reward_vals), list(alphas), list(betas))])

  rewards = {
    agent: rew for agent, rew in zip(list(rewards.keys()), new_reward_vals)
  }

  return rewards, next_e



class MeltingPotEnv(multi_agent_env.MultiAgentEnv):
  """Interfacing Melting Pot substrates and RLLib MultiAgentEnv."""

  def __init__(self, env: dmlab2d.Environment, alphas, betas, gamma, lmbda):
    """Initializes the instance.

    Args:
      env: dmlab2d environment to wrap. Will be closed when this wrapper closes.
    """
    self._alphas = alphas
    self._betas = betas
    self._gamma = gamma # discount rate
    self._lmbda = lmbda # new hparam


    self._env = env
    self._num_players = len(self._env.observation_spec())
    self._ordered_agent_ids = [
        PLAYER_STR_FORMAT.format(index=index)
        for index in range(self._num_players)
    ]
    # RLLib requires environments to have the following member variables:
    # observation_space, action_space, and _agent_ids
    self._agent_ids = set(self._ordered_agent_ids)
    
    # RLLib expects a dictionary of agent_id to observation or action,
    # Melting Pot uses a tuple, so we convert them here
    self.observation_space = self._convert_spaces_tuple_to_dict(
        utils.spec_to_space(self._env.observation_spec()),
        remove_world_observations=True)
    self.action_space = self._convert_spaces_tuple_to_dict(
        utils.spec_to_space(self._env.action_spec()))
    
    # init smoothed reward
    self.cur_e = np.zeros((self._num_players,))

    super().__init__()

  def reset(self, *args, **kwargs):
    """See base class."""
    timestep = self._env.reset()
    return utils.timestep_to_observations(timestep), {}

  def step(self, action_dict):
    """See base class."""
    actions = [action_dict[agent_id] for agent_id in self._ordered_agent_ids]
    timestep = self._env.step(actions)
    rewards = {
        agent_id: timestep.reward[index]
        for index, agent_id in enumerate(self._ordered_agent_ids)
    }
    rewards, self.cur_e = rearrange_IA(rewards, self.alphas, self.betas, self.gamma, self.lmbda)
    done = {'__all__': timestep.last()}
    info = {}

    observations = utils.timestep_to_observations(timestep)
    return observations, rewards, done, done, info

  def close(self):
    """See base class."""

    self._env.close()

  def get_dmlab2d_env(self):
    """Returns the underlying DM Lab2D environment."""

    return self._env

  # Metadata is required by the gym `Env` class that we are extending, to show
  # which modes the `render` method supports.
  metadata = {'render.modes': ['rgb_array']}

  def render(self) -> np.ndarray:
    """Render the environment.

    This allows you to set `record_env` in your training config, to record
    videos of gameplay.

    Returns:
        np.ndarray: This returns a numpy.ndarray with shape (x, y, 3),
        representing RGB values for an x-by-y pixel image, suitable for turning
        into a video.
    """

    observation = self._env.observation()
    world_rgb = observation[0]['WORLD.RGB']

    # RGB mode is used for recording videos
    return world_rgb

  def _convert_spaces_tuple_to_dict(
      self,
      input_tuple: spaces.Tuple,
      remove_world_observations: bool = False) -> spaces.Dict:
    """Returns spaces tuple converted to a dictionary.

    Args:
      input_tuple: tuple to convert.
      remove_world_observations: If True will remove non-player observations.
    """

    return spaces.Dict({
        agent_id: (utils.remove_unrequired_observations_from_space(input_tuple[i])
                   if remove_world_observations else input_tuple[i])
        for i, agent_id in enumerate(self._ordered_agent_ids)
    })