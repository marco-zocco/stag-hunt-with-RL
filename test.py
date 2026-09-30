import gymnasium as gym
import gymnasium_stag_hunt
import time

env = gym.make("StagHunt-Hunt-v0", obs_type='image', ) 
env.reset()
for iteration in range(1000):
    time.sleep(.2)
    obs, rewards, done, info = env.step([env.action_space.sample(), env.action_space.sample()])
    env.render()    
env.close()