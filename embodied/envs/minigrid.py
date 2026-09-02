import gymnasium
from gymnasium import spaces
from minigrid.wrappers import (
    ImgObsWrapper,
    RGBImgObsWrapper,
    RGBImgPartialObsWrapper,
)

from embodied.core.wrappers import ResizeImage
from embodied.envs.from_gymnasium import FromGymnasium


class HideMission(gymnasium.ObservationWrapper):

  def __init__(self, env):
    super().__init__(env)
    assert isinstance(self.observation_space, spaces.Dict)
    self.observation_space = spaces.Dict({
        key: value for key, value in self.observation_space.spaces.items()
        if key != 'mission'
    })

  def observation(self, observation):
    observation = dict(observation)
    observation.pop('mission', None)
    return observation


class WrappedMinigrid(FromGymnasium):

  def __init__(
      self,
      task,
      fully_observable=False,
      hide_mission=True,
      max_episode_steps=500,
      seed=None,
  ):
    env_id = task if task.startswith('MiniGrid-') else f'MiniGrid-{task}'
    env = gymnasium.make(
        env_id,
        render_mode='rgb_array',
        max_episode_steps=max_episode_steps,
    )
    if fully_observable:
      env = RGBImgObsWrapper(env)
      if hide_mission:
        env = HideMission(env)
    else:
      env = ImgObsWrapper(RGBImgPartialObsWrapper(env))
    super().__init__(env, seed=seed)


class Minigrid(ResizeImage):

  def __init__(self, task, size=(64, 64), **kwargs):
    super().__init__(WrappedMinigrid(task, **kwargs), size=tuple(size))
