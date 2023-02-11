import game
import numpy as np

 
game = game.Game("dqn_agent", "cuda")


hyperparameter = {
  "lr_start": 1e-4,
  "lr_end": 1e-4,
  "batch_size": 128,
  "gamma": 0.9,
  "eps_start": 0.9,
  "eps_end": 1e-3
}
    

game.train_agent(False, 100, 100, hyperparameter)  
score = game.main(True)
print("Score: ", score)


#Notice:
#Choose smoothL1Loss not MSE -> gradients too high -> probably gradient clipping
#Choose gamma carfully not too high not too low -> about 0.8-0.9 ->1/(1-gamma) timesteps related
#Normalize input

#Choose high rewards if using sparse ones (or not works both)
#give reward for staying alive (but a small one -> relative to bad rewards)