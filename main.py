
import gym
import envs.pick

env = gym.make("MultiGrid-Color-Gather-Env-8x8-v0", color_pick="green")
# pre-training phase
num_tasks = 5
num_episodes = 500
obs = env.reset()

for task_num in range(num_tasks):
    for episode_num in range(num_episodes):
        action = env.action_space.sample()
        env.render(mode='human')
        next_obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
            done = False

env.close()