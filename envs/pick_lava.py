"""Implements the multi-agent gather environments.

The agents must pick up (move on top of) items in the environment.
"""
import berrygrid.minigrid as minigrid
import numpy as np
import berrygrid.multigrid as multigrid
from berrygrid.register import register
import berrygrid.minigrid as minigrid
from berrygrid.minigrid import Water


class WaterPickEnv(multigrid.MultiGridEnv):
    """Object gathering environment."""

    def __init__(self,
                 color_pick="red",
                 place_water=False,
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
        self.place_water = place_water
        self.size = size
        self.strip2_row = 5
        if n_colors > len(minigrid.IDX_TO_COLOR):
            raise ValueError('Too many colors requested')

        self.collected_colors = [0] * n_colors
        self.battery_enabled = True
        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            n_agents=n_agents,
            fully_observed=True)
        self.metrics = {'max_gathered': 0, 'other_gathered': 0}

    def reset(self, color="red", water=False):
        self.color_pick = color
        self.place_water = water
        self.collected_colors = [0] * self.n_colors
        return super(WaterPickEnv, self).reset()

    def _gen_grid(self, width, height):
        self.grid = multigrid.Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.objects = []
        self.colors = list(range(len(minigrid.IDX_TO_COLOR)))

        if self.battery_enabled:
            # Set Battery Location
            battery = minigrid.Battery()
            pos_battery = [4, self.size - 2]
            self.grid.set(pos_battery[0], pos_battery[1], battery)

            if battery is not None:
                battery.init_pos = pos_battery
                battery.cur_pos = pos_battery

        # Set Single Goal Location for Agent
        for i in range(3):
            if self.random_colors:
                color_pick = minigrid.IDX_TO_COLOR[np.random.choice(self.colors)]
            else:
                color_pick = self.color_pick
            self.objects.append(minigrid.Ball(color=color_pick))
            self.place_obj(self.objects[-1], max_tries=100)

        for _ in range(self.n_clutter):
            self.place_obj(minigrid.Wall(), max_tries=100)

        # Set Dummy Goal Locations
        remaining_colors = [i for i in self.colors if i != minigrid.COLOR_TO_IDX[self.color_pick]]
        remaining_colors = remaining_colors[:self.n_colors - 1]
        for i in range(self.width - 6):
            self.grid.set(3 + i, 1, Water())
            self.grid.set(3 + i, self.strip2_row, Water())

        for j in remaining_colors:
            for i in range(3):
                if self.random_colors:
                    color = minigrid.IDX_TO_COLOR[np.random.choice(remaining_colors)]
                else:
                    color = minigrid.IDX_TO_COLOR[j]
                self.objects.append(minigrid.Ball(color=color))
                self.place_obj(self.objects[-1], max_tries=100)

        self.place_agent()
        self.mission = 'pick up objects'

    def step(self, action):
        obs, _, done, info = multigrid.MultiGridEnv.step(self, action)
        reward = [0] * self.n_agents
        for i, obj in enumerate(self.carrying):
            if obj:
                color_idx = self.colors.index(minigrid.COLOR_TO_IDX[obj.color])
                self.collected_colors[color_idx] += 1
                if obj.color == self.color_pick:
                    reward[i] += 1
                    self.metrics['max_gathered'] += 1
                    self.place_obj(obj, max_tries=100)
                else:
                    reward[i] += 0
                    self.metrics['other_gathered'] += 1
                    self.place_obj(obj, max_tries=100)

                self.carrying[i] = None
        return obs, reward, done, info


class RandomLavaColorGatherEnv8x8(WaterPickEnv):
    def __init__(self, **kwargs):
        super().__init__(size=12
                         , n_agents=1, n_goals=1, extra_goals=6, n_clutter=0, n_colors=7, **kwargs)


if hasattr(__loader__, 'name'):
    module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
    module_path = __loader__.fullname

register(
    env_id='MultiGrid-LavaColor-Gather-Env-8x8-v0',
    entry_point=module_path + ':RandomLavaColorGatherEnv8x8')
