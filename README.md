# Banana collector project: p1 in DRLND 
---
## Project details: 

This project is being done as part of the Udacity Deep Reinforcement Learning Nanodegree, a four month course that I am taking. The goal of this project is to train an agent to navigate a large square environment collecting as many yellow bananas as possible while avoiding blue bananas.
<p align="center"><img src="https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif" alt="Example game of isolation" width="50%" style="middle"></p>

## Environment details: 

The environment is episodic and has a single agent, a continous state space and a discrete action space. Following are the details of the environment.
Unity brain name: BananaBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 37
        Number of stacked Vector Observation: 1
        Vector Action space type: discrete
        Vector Action space size (per agent): 4
        Vector Action descriptions: , , , 

 We are working with a state space of 37 dimensions that it contains the agent's velocity and an action space of size 4 corresponding to 
 - `0` - for moving forward
 - `1` - for moving backward
 - `2` - for turning left 
-  `3` - for turning right
 The environment is considered solved when the agent get an average score of +13 over 100 consecutive episodes.

## Getting started

1. Clone this repo using `git clone https://github.com/RihabGorsan/NavigationVanillaDQN.git` </br>
2. Begin by importing the necessary packages. So first create a conda environment  </br>
`conda create --name navigation python=3.6` </br>
and then install the following packages: </br>
`conda install -y pytorch -c pytorch` </br>
`pip install unityagents==0.4.0 ` </br>
`pip install mlagents` </br>

3. Download the Banana unity environment 
You can download the environment form one  of the links below. Just please select the enviornment that matches your OS

    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Unzip the file hen place it in the cloned project. 


## Train your agent
To train the agent run [`python dqn.py`](dqn.py) or the [`Navigation.ipynb`](Navigation.ipynb) notebook.  This will fire up the Unity environment and output live training statistics to the command line.  When training is finished you'll have a saved model in `checkpoint.pth`.

To watch your trained agent interact with the environment run `python dqn.py`, just set `train` parameter to False and assign the path of the trained weights to the `filename` parameter.  This will load the saved weights from a checkpoint file.  A previously trained model is included in this repo and named `saved_model_weights.

And following are the results of the training:
```
Episode 100     Average Score: 4.49 
Episode 200     Average Score: 8.02 
Episode 300     Average Score: 11.58 
Episode 346     Average Score: 13.02 
Environment solved in 346 episodes!     Average Score: 13.02
```

Feel free to experiment with modifying the hyperparameters to see how it affects training:

- [model.py](model.py) : you can change the architecture of the network.
- [agent.py](agent.py) : play with the hyperparams of an RL agent like gamma, epsilon, tau ..

## Report
See the [report](report.md) for more details.  


