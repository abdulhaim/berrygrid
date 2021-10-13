"""Implements the multi-agent gather environments.

The agents must pick up (move on top of) items in the environment.
"""
import minigrid
import numpy as np
import multigrid
from register import register


class PickEnv(multigrid.MultiGridEnv):
  """Object gathering environment."""

  def __init__(self,
               color_pick="red",
               size=15,
               n_agents=2,
               n_goals=1,
               n_clutter=0,
               n_colors=2,
               extra_goals=4,
               random_colors=False,
               max_steps=250,
               **kwargs):
    """Constructor for multi-agent gridworld environment generator.

    Args:
      size: Number of tiles for the width and height of the square grid.
      n_agents: The number of agents playing in the world.
      n_goals: The number of coins in the environment.
      n_clutter: The number of blocking objects in the environment.
      n_colors: The number of different object colors.
      random_colors: If true, each color has a random number of coins assigned.
      max_steps: Number of environment steps before the episode end (max episode
        length).
      **kwargs: See superclass.
    """
    self.n_clutter = n_clutter
    self.n_goals = n_goals
    self.n_colors = n_colors
    self.random_colors = random_colors
    self.extra_goals = extra_goals
    self.color_pick = color_pick
    if n_colors >= len(minigrid.IDX_TO_COLOR):
      raise ValueError('Too many colors requested')

    self.collected_colors = [0] * n_colors
    super().__init__(
        grid_size=size,
        max_steps=max_steps,
        n_agents=n_agents,
        fully_observed=True,
        **kwargs)
    self.metrics = {'max_gathered': 0, 'other_gathered': 0}

  def reset(self):
    self.collected_colors = [0] * self.n_colors
    return super(PickEnv, self).reset()

  def _gen_grid(self, width, height):
    self.grid = multigrid.Grid(width, height)
    self.grid.wall_rect(0, 0, width, height)
    self.objects = []
    self.colors = list(range(len(minigrid.IDX_TO_COLOR)))

    # Set Battery Location
    battery = minigrid.Battery()
    pos_battery = [4,6]
    self.grid.set(pos_battery[0], pos_battery[1], battery)

    if battery is not None:
      battery.init_pos = pos_battery
      battery.cur_pos = pos_battery

    # Set Single Goal Location for Agent
    for i in range(self.n_goals):
      if self.random_colors:
        color = minigrid.IDX_TO_COLOR[np.random.choice(self.colors)]
      else:
        # color = minigrid.IDX_TO_COLOR[self.colors[i % self.n_colors]]
        color = self.color_pick
      self.objects.append(minigrid.Ball(color=color))
      self.place_obj(self.objects[i], max_tries=100)
    for _ in range(self.n_clutter):
      self.place_obj(minigrid.Wall(), max_tries=100)

    # Set Dummy Goal Locations
    remaining_colors = [i for i in self.colors if i!=minigrid.COLOR_TO_IDX[self.color_pick]]
    for i in range(1, self.extra_goals):
      if self.random_colors:
        color = minigrid.IDX_TO_COLOR[np.random.choice(self.colors)]
      else:
        color = minigrid.IDX_TO_COLOR[remaining_colors[i % self.n_colors-2]]
      self.objects.append(minigrid.Ball(color=color))
      self.place_obj(self.objects[i], max_tries=100)

    self.place_agent()

    self.mission = 'pick up objects'

  def step(self, action):
    obs, _, done, info = multigrid.MultiGridEnv.step(self, action)
    reward = [0] * self.n_agents
    for i, obj in enumerate(self.carrying):
      if obj:
        color_idx = self.colors.index(minigrid.COLOR_TO_IDX[obj.color])

        # self.collected_colors[color_idx] += 1
        # if max(self.collected_colors) == self.collected_colors[color_idx]:
        if color_idx == i:
          reward[i] += 1
          self.metrics['max_gathered'] += 1
          self.place_obj(obj, max_tries=100)
        else:
          self.metrics['other_gathered'] += 1
        self.carrying[i] = None
    return obs, reward, done, info


class RandomColorGatherEnv8x8(PickEnv):
  def __init__(self, **kwargs):
    super().__init__(size=8, n_agents=1, n_goals=1, extra_goals=4, n_clutter=0, n_colors=5, **kwargs)

if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

register(
    env_id='MultiGrid-Color-Gather-Env-8x8-v0',
    entry_point=module_path + ':RandomColorGatherEnv8x8')

