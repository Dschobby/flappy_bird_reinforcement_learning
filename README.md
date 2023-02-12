# Flappy bird reinforcement learning agent
This code provides an environment for the game flappy bird. Using this environment one can train an deep-q network (DQN) agent for playing the game using reinforcement learning. Additionally one can choose freely between the following agents:
- User agent: User played game (bump bird with space)
- Random agent: Agent which performs random actions
- DQN agent: Agent using a deep Q-Network for performing actions

The game and its agent is run by initializing with
```python
game = game.Game(agent_name = "dqn_agent", device = "cpu")
```
To run a game execute the main function
```python
game.main(draw = "True")
```
If using a DQN agent one can train with
```python
game.train_agent(draw = "False", episodes = 100, batches = 100, hyperparameters)
```

## Deep Q-learning setup
The game environment returns three features as input for the DQN agent:
- Horizontal distance to next pipe
- Vertical distance to lower next pipe
- Speed of bird

The returned reward from the environment after performing an action is:
- 0.1 for surviving 
- -10 for colliding

The agent has two possible actions:
- bump the bird
- doing nothing

The whole states/rewards/actions are stored in an experience buffer of the agent used for training after each episode.<br />
The DQN agents architecture is a neural network with one hidden layer of size 128.

## Trained DQN agent example
Example of trained DQN agent using $\gamma$
![](https://github.com/Dschobby/flappy_bird_reinforcement_learning/blob/main/animations/flappy_bird_animation.gif)

## Required packages
- numpy
- pytorch
- pygame
- tqdm
