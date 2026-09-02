import functools

import elements
import embodied
import gymnasium
import numpy as np


class FromGymnasium(embodied.Env):

  def __init__(
      self, env, obs_key='image', act_key='action', seed=None, **kwargs):
    if isinstance(env, str):
      kwargs.setdefault('render_mode', 'rgb_array')
      self._env = gymnasium.make(env, **kwargs)
    else:
      assert not kwargs, kwargs
      self._env = env
    self._obs_dict = isinstance(
        self._env.observation_space, gymnasium.spaces.Dict)
    self._act_dict = isinstance(
        self._env.action_space, gymnasium.spaces.Dict)
    self._obs_key = obs_key
    self._act_key = act_key
    self._seed = seed
    self._done = True
    self._info = None

  @property
  def env(self):
    return self._env

  @property
  def info(self):
    return self._info

  @functools.cached_property
  def obs_space(self):
    if self._obs_dict:
      spaces = self._flatten(self._env.observation_space.spaces)
    else:
      spaces = {self._obs_key: self._env.observation_space}
    spaces = {key: self._convert(value) for key, value in spaces.items()}
    return {
        **spaces,
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }

  @functools.cached_property
  def act_space(self):
    if self._act_dict:
      spaces = self._flatten(self._env.action_space.spaces)
    else:
      spaces = {self._act_key: self._env.action_space}
    spaces = {key: self._convert(value) for key, value in spaces.items()}
    spaces['reset'] = elements.Space(bool)
    return spaces

  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      obs, self._info = self._env.reset(seed=self._seed)
      self._seed = None
      return self._obs(obs, 0.0, is_first=True)
    if self._act_dict:
      action = self._unflatten(action)
      action.pop('reset', None)
    else:
      action = action[self._act_key]
    obs, reward, terminated, truncated, self._info = self._env.step(action)
    self._done = bool(terminated or truncated)
    return self._obs(
        obs,
        reward,
        is_last=self._done,
        is_terminal=bool(self._info.get('is_terminal', terminated)),
    )

  def _obs(
      self, obs, reward, is_first=False, is_last=False, is_terminal=False):
    if not self._obs_dict:
      obs = {self._obs_key: obs}
    obs = self._flatten(obs)
    obs = {key: np.asarray(value) for key, value in obs.items()}
    obs.update(
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
    )
    return obs

  def render(self):
    image = self._env.render()
    assert image is not None
    return image

  def close(self):
    try:
      self._env.close()
    except Exception:
      pass

  def _flatten(self, nest, prefix=None):
    result = {}
    for key, value in nest.items():
      key = prefix + '/' + key if prefix else key
      if isinstance(value, gymnasium.spaces.Dict):
        value = value.spaces
      if isinstance(value, dict):
        result.update(self._flatten(value, key))
      else:
        result[key] = value
    return result

  def _unflatten(self, flat):
    result = {}
    for key, value in flat.items():
      parts = key.split('/')
      node = result
      for part in parts[:-1]:
        node = node.setdefault(part, {})
      node[parts[-1]] = value
    return result

  def _convert(self, space):
    if isinstance(space, gymnasium.spaces.Discrete):
      assert space.start == 0, space
      return elements.Space(np.int32, (), 0, space.n)
    if isinstance(space, gymnasium.spaces.MultiDiscrete):
      assert np.all(space.start == 0), space
      return elements.Space(np.int32, space.shape, 0, space.nvec)
    if isinstance(space, gymnasium.spaces.MultiBinary):
      return elements.Space(np.int32, space.shape, 0, 2)
    if isinstance(space, gymnasium.spaces.Box):
      return elements.Space(space.dtype, space.shape, space.low, space.high)
    raise NotImplementedError(
        f'Unsupported Gymnasium space: {type(space).__name__}')
