from config import *
import pygame
from pygame.locals import *



"""
Class controlling gameobject Bird
"""
class Bird:

    def __init__(self, image):

        #Image of bird
        self.image =  image

        #Vertical speed of bird
        self.speed = SPEED

        #Position of bird
        self.pos = self.image.get_rect() # Position # left,top, width, height

        #Set position of bird
        self.pos[0] = round(SCREEN_WIDHT / 6 / GAME_SPEED) * GAME_SPEED #ensure divisible trough GAME_SPEED -> used for score function
        self.pos[1] = SCREEN_HEIGHT / 2

    def update(self):
        self.speed += GRAVITY
        self.pos[1] += self.speed

    def bump(self):
        self.speed = -SPEED

    def draw(self, screen):
        screen.blit(self.image, self.pos)


"""
Class controlling gameobject Pipe
"""
class Pipe:

    def __init__(self, image, inverted, xpos, ysize):

        #Image of pipe
        self.image = image

        #Position of pipe
        self.pos = self.image.get_rect()

        #Set pipe position
        self.inverted = inverted
        self.pos[0] = xpos
        if inverted:
            self.image = pygame.transform.flip(self.image, False, True)
            self.pos[1] = - (self.pos[3] - ysize)
        else:
            self.pos[1] = SCREEN_HEIGHT - ysize

    def update(self):
        self.pos[0] -= GAME_SPEED
        # if pos kleiner 1 delete itself

    def draw(self, screen):
        screen.blit(self.image, self.pos)

        
"""
Class controlling gameobject Ground
"""
class Ground:
    
    def __init__(self, image, xpos):

        #Image of ground
        self.image = image

        #Position of ground
        self.pos = self.image.get_rect()

        #Set position of ground
        self.pos[0] = xpos
        self.pos[1] = SCREEN_HEIGHT - GROUND_HEIGHT

    def draw(self, screen):
        screen.blit(self.image, self.pos)
