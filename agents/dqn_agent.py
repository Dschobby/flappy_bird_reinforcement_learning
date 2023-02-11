import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


"""
Neural Network model structure for the DQN agent
"""
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(3, 128)
        self.linear2 = nn.Linear(128, 2)

    def forward(self, x):
        y = self.linear2(F.relu(self.linear1(x)))
        return y


"""
DQN agent for reinforcement learning
"""
class DQN_agent:
    def __init__(self, device):
        #Set up model
        self.device = device
        self.model = NeuralNetwork().to(device)

        #Set up experience buffer
        self.buffer = [[],[],[],[]]
        self.buffer_size = 10000

        #Training hyperparameters
        self.lr = 0.001
        self.batch_size = 64
        self.gamma = 0.2
        self.epsilon = 0.0

        #Set up training helpers
        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def act(self, state, train):
        state = torch.Tensor(state).to(self.device)

        #Choose action with max q value of if in training mode choose action according to epsilon greedy policy
        if np.random.rand() > self.epsilon or not train:
            with torch.no_grad():
                return self.model(state).data.max(0)[1].detach().cpu()
        else:
            return torch.randint(2,(1,))[0]

    def train(self):
        #Check buffer size
        if len(self.buffer[0]) < self.batch_size:
            return 
        #Delete old data from buffer if buffer size is surpassed
        if len(self.buffer[0]) > self.buffer_size:
            del self.buffer[0][0:len(self.buffer[0])-self.buffer_size]
            del self.buffer[1][0:len(self.buffer[1])-self.buffer_size]
            del self.buffer[2][0:len(self.buffer[2])-self.buffer_size]
            del self.buffer[3][0:len(self.buffer[3])-self.buffer_size]
        
        #Create data batch
        batch_ind = torch.randperm(len(self.buffer[0]))[0:self.batch_size]
        batch_state = torch.Tensor(self.buffer[0])[batch_ind].to(self.device)
        batch_next_state = torch.Tensor(self.buffer[1])[batch_ind].to(self.device)
        batch_action = torch.Tensor([self.buffer[3]]).long().reshape(len(self.buffer[3]),-1)[batch_ind].to(self.device)
        batch_reward = torch.Tensor(self.buffer[2])[batch_ind].to(self.device)

        #Calculate expected and current q-value
        current_q_values = self.model(batch_state).gather(1,batch_action)
        with torch.no_grad():
            max_next_q_values = self.model(batch_next_state).detach().max(1)[0]
        expected_q_values = batch_reward + (self.gamma * max_next_q_values)

        #loss
        loss = self.loss_fn(current_q_values, expected_q_values.unsqueeze(1))

        #backpropagation of loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 100) # gradient clipping for preventing exploding gradients
        self.optimizer.step()

        return loss.item()



