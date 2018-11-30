from unityagents import UnityEnvironment
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
#%matplotlib inline

from agent import Agent

path="/home/rihab/Documents/bananaproject/Banana_Linux/Banana.x86_64"

#Load the banana environment
env = UnityEnvironment(file_name=path)
#get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
#get action and state sizes
action_size = brain.vector_action_space_size
state_size = 37

 #instanciate the agent
agent = Agent(state_size=state_size, action_size=action_size, seed=0)


def dqn(n_episodes=1000, max_t=1000, eps_start=0.5, eps_end=0.01, eps_decay=0.99):
    """
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        eps = eps / i_episode
        env_info = env.reset(train_mode=True)[brain_name]            #reset the environment
        state= env_info.vector_observations[0] 
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)                           #select an action
            env_info = env.step(action)[brain_name]                  #send action to the enviornment
            next_state = env_info.vector_observations[0]             #get the next state
            reward = env_info.rewards[0]                             #get th reward
            done= env_info.local_done[0]                             #if episode has  finished
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        #eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores

scores = dqn()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()