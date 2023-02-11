import pygame
from pygame.locals import *


class User_agent:
    def __init__(self):
        pass

    def act(self, state, train):
        for event in pygame.event.get():
            if event.type == QUIT:
                return -1
            if event.type == KEYDOWN:
                if event.key == K_SPACE or event.key == K_UP:
                    return 1
        return 0