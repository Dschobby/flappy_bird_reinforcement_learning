from config import *
import objects
import pygame
from pygame.locals import *
import random
import sys
import time
import numpy as np
import torch
import agents.user_agent, agents.random_agent, agents.dqn_agent



#Agents
AGENTS = ["user_agent", "random_agent", "dqn_agent"]

#Pygame image loading
bird_image = pygame.image.load('assets/bluebird-upflap.png')
bird_image = pygame.transform.scale(bird_image, (BIRD_WIDTH, BIRD_HEIGHT))
pipe_image = pygame.image.load('assets/pipe-green.png')
pipe_image = pygame.transform.scale(pipe_image, (PIPE_WIDHT, PIPE_HEIGHT))
ground_image = pygame.image.load('assets/base.png')
ground_image = pygame.transform.scale(ground_image, (GROUND_WIDHT, GROUND_HEIGHT))
BACKGROUND = pygame.image.load('assets/background-day.png')
BACKGROUND = pygame.transform.scale(BACKGROUND, (SCREEN_WIDHT, SCREEN_HEIGHT))



"""
Main game class which is running and controlling the game
"""
class Game:
    
    def __init__(self, agent_name, device):

        #Initialize agent
        if not agent_name in AGENTS: sys.exit("Agent not defined")
        if device != "cpu" and device != "cuda": sys.exit("Computing device not available")
        if agent_name == "user_agent": 
            self.agent = agents.user_agent.User_agent()
            print("Initialize game with: User_agent")
        if agent_name == "random_agent": 
            self.agent = agents.random_agent.Random_agent()
            print("Initialize game with: Random_agent")
        if agent_name == "dqn_agent": 
            self.agent = agents.dqn_agent.DQN_agent(device)
            print("Initialize game with: DQN_agent")
            print("Trainable parameters: {}".format(sum(p.numel() for p in vars(self.agent)["model"].parameters())))
        self.device = device

        #Game objects (Get initialized new every game played)
        self.bird = None
        self.ground = None
        self.pipes = None
        self.score = None
        self.turn = None

        #Training mode for agent
        self.train = False

    def init_game(self):

        #Initialize game objects
        self.bird = objects.Bird(bird_image)
        self.ground = objects.Ground(ground_image, 0)
        self.pipes = []
        self.score = 0
        self.turn = 0

        #Initialize pipes
        for i in range(3):
            #Pipe initial positions
            xpos = PIPE_DISTANCE * i + PIPE_DISTANCE
            ysize = random.randint(200, 300)

            #Append pipes to list
            self.pipes.append(objects.Pipe(pipe_image, False, xpos, ysize))
            self.pipes.append(objects.Pipe(pipe_image, True, xpos, SCREEN_HEIGHT - ysize - PIPE_GAP))

    def pipe_handling(self):
        #if pipes out of screen add new ones and remove old
        if vars(self.pipes[0])["pos"][0] <= -100:

            #Remove old pipes
            del self.pipes[0]
            del self.pipes[0]

            #New pipe initial positions
            xpos = PIPE_DISTANCE * 3 - 100
            ysize = random.randint(150, 350)

            #Append new pipes
            self.pipes.append(objects.Pipe(pipe_image, False, xpos, ysize))
            self.pipes.append(objects.Pipe(pipe_image, True, xpos, SCREEN_HEIGHT - ysize - PIPE_GAP))
                
    def collision(self):
        #Check ground and roof collision
        if vars(self.bird)["pos"][1] < 0 or vars(self.bird)["pos"][1] > SCREEN_HEIGHT - GROUND_HEIGHT - BIRD_HEIGHT:
            return True

        #Check for pipe collision
        if vars(self.pipes[0])["pos"][0] - vars(self.bird)["pos"][2] < vars(self.bird)["pos"][0] and vars(self.bird)["pos"][0] < vars(self.pipes[0])["pos"][0] +vars(self.pipes[0])["pos"][2]:
            if vars(self.pipes[0])["pos"][1] < vars(self.bird)["pos"][1] + vars(self.bird)["pos"][3] or vars(self.pipes[0])["pos"][1] - PIPE_GAP > vars(self.bird)["pos"][1]:
                return True

        return False

    def score_update(self):
        if vars(self.bird)["pos"][0] == vars(self.pipes[0])["pos"][0]:
            self.score += 1

    def game_state(self):
        state = []

        #Gamestate passing to the agent: 1-horizontal distance to next pipe, 2-vertical distance to lower next pipe, 3-bird speed
        for pipe in self.pipes:
            if vars(self.bird)["pos"][0] < vars(pipe)["pos"][0] + vars(pipe)["pos"][2]: #Check which pipe is the next one
                state.append((- vars(self.bird)["pos"][0] + vars(pipe)["pos"][2] + vars(pipe)["pos"][0]) / PIPE_DISTANCE)
                state.append((vars(pipe)["pos"][1] - PIPE_GAP/2 - vars(self.bird)["pos"][1] - vars(self.bird)["pos"][3] / 2) / SCREEN_HEIGHT * 2)
                break
        state.append(vars(self.bird)["speed"] / SPEED)

        return state

    def reward(self):

        reward = 0.1 #reward of 0.1 for surviving
        if self.collision():
            reward = -10 #reward -10 for colliding

        return round(reward,4)

    def main(self, draw): 

        #Initialize pygame screen if wanted
        if draw:
            pygame.init()
            screen = pygame.display.set_mode((SCREEN_WIDHT, SCREEN_HEIGHT))
            pygame.display.set_icon(bird_image)
            pygame.display.set_caption('Flappy Bird')
            clock = pygame.time.Clock()

        #Initialize game
        active_episode = True
        self.init_game()

        #Game loop
        while active_episode:

            if draw:
                clock.tick(30)
                screen.blit(BACKGROUND, (0, 0))

                #Check for closing game window
                if not isinstance(self.agent, agents.user_agent.User_agent):
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            active_episode = False

            #Get and execute agent action
            state = self.game_state()
            action = self.agent.act(state, self.train)
            if action == 1: self.bird.bump()
            if action == -1: active_episode = False

            #Updating environment
            self.bird.update()
            for pipe in self.pipes: pipe.update()
            self.score_update()
            
            #Remove pipes if out of screen and instantiate new ones
            self.pipe_handling()
            
            #Check for collisions
            if self.collision(): active_episode = False

            #Give state to experience buffer if in training mode of dqn_agent
            if self.train:
                vars(self.agent)["buffer"][0].append(state)
                vars(self.agent)["buffer"][1].append(self.game_state())
                vars(self.agent)["buffer"][2].append(self.reward())
                if action == 0: vars(self.agent)["buffer"][3].append(torch.Tensor([0]))
                if action == 1: vars(self.agent)["buffer"][3].append(torch.Tensor([1]))
            self.turn += 1
                
            #Update screen
            if draw:
                self.bird.draw(screen)
                self.ground.draw(screen)
                for pipe in self.pipes: pipe.draw(screen)

                pygame.display.update()

            #Terminate episode after reaching score of 100
            if self.score >= 100:
                active_episode = False

        #Quit pygame window
        if draw:
            pygame.display.quit()
            pygame.quit()

        return self.score

        
    def train_agent(self, draw, episodes, batches, hyperparameter):
        
        #Training control parameters
        convergence = 0 #parameter controlling if convergence happened
        loss = 0
        mean_score = []
        time_start = time.time()

        #Print training initials
        print("Start training process of agent")
        if self.device == "cuda": print("Using {} device".format(self.device), ": ", torch.cuda.get_device_name(0))
        else: print("Using {} device".format(self.device))
        print("Used training hyperparameters: ",hyperparameter)

        #Check if agent is trainable
        if not isinstance(self.agent, agents.dqn_agent.DQN_agent):
            sys.exit("Agent is not trainable")

        self.train = True

        for episode in range(1, episodes + 1):

            #Specify episode lr and epsilon
            eps = hyperparameter["eps_end"] + (hyperparameter["eps_start"] - hyperparameter["eps_end"]) * np.exp(-1. * episode /episodes * 10)
            lr = hyperparameter["lr_end"] + (hyperparameter["lr_start"] - hyperparameter["lr_end"]) * np.exp(-1. * episode /episodes * 10)
            vars(self.agent)["lr"] = lr
            vars(self.agent)["batch_size"] = hyperparameter["batch_size"]
            vars(self.agent)["gamma"] = hyperparameter["gamma"]
            vars(self.agent)["epsilon"] = eps

            #Run an episode
            _ = self.main(draw)
            
            #Train agent
            for i in range(batches):
                loss += self.agent.train()

            #Test agent
            self.train = False
            test_score = self.main(False)
            mean_score.append(test_score)
            if test_score == 100: convergence += 1 #look if agent has perfectly performed the game
            else: convergence = 0
            
            #Print training perfomance log
            time_step = time.time()
            if episode % 10 == 0 or convergence == 2: 
                print("Episode: [{}/{}]".format(episode, episodes) + 
                    "    -Time: [{}<{}]".format(time.strftime("%M:%S", time.gmtime(time_step-time_start)), time.strftime("%M:%S", time.gmtime((time_step-time_start) * episodes/episode))) +
                    " {}s/it".format(round((time_step-time_start)/episode,1)) +
                    "    -Loss: {}".format(round(loss/batches,6)) + 
                    "    -MeanTestScore: {}".format(round(np.mean(mean_score[-2:]))))
                mean_score = []
            loss = 0

            #Terminate training if agent never collides after two training procedures in a row
            if convergence == 2: 
                print("Agent performed faultless")
                break
            self.train = True


        self.train = False
        
        print("Training finished after {} episodes".format(episode))
