# Flappy bird reinforcement learning agent
This code provides an environment for the game flappy. The game can be run using different agents:
- User agent: User played game
- Random agent: Agent which performs random actions
- DQN agent: Agent using a deep Q-Network for performing actions
Additionally the code provides a procedure for training the DQN agent.

The game and its agent is run by initializing with
```python
game = game.Game("agent_name", "computing_device")
```
To run a game execute the main function
```python
game.main("draw_game")
```
If using a DQN agent one can train with
```python
game.train_agent("draw_game", episodes, batches_per_episode, hyperparameters)
```

## Deep Q-learning setup
The game environment returns three features as input for the DQN agent:
- Horizontal distance to next pipe
- Vertical distance to lower next pipe
- Speed of bird

The returned reward from the environment after performing an action is:
- 0.1 for surviving 
- -10 for colliding

The whole states/rewards are stored in an experience buffer of the agent used for training after each episode.<br />
The DQN agents architecture is a neural network with one hidden layer of size 128.

## Trained DQN agent example
![](https://github.com/Dschobby/flappy_bird_reinforcement_learning/blob/main/animations/flappy_bird_animation.gif)

## Required packages
- numpy
- pytorch
- pygame
- tqdm
