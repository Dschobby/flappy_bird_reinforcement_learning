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

The whole states/rewards/actions are stored in an experience buffer of the agent used for training after each episode. During the training procedure the agent uses the $\epsilon$-greedy policy with decreasing $\epsilon$.<br />
The DQN agents architecture is a neural network with one hidden layer of size 128.

## Trained DQN agent example
Example of trained DQN agent using a training of 100 episodes each with 100 batches of size 128 and learning rate of $\tau = 1e^{-4}$ with hyperparameters $\gamma = 0.8$, $\epsilon_s = 0.9$, $\epsilon_e = 1e^{-2}$. <br /><br />
![](https://github.com/Dschobby/flappy_bird_reinforcement_learning/blob/main/animations/flappy_bird_animation.gif)

## Required packages
- numpy
- pytorch
- pygame
