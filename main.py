import game
import numpy as np

 
game = game.Game("dqn_agent", "cuda")


hyperparameter = {
  "lr_start": 1e-4,
  "lr_end": 1e-4,
  "batch_size": 128,
  "gamma": 0.9,
  "eps_start": 0.9,
  "eps_end": 1e-2
}
    

game.train_agent(False, 100, 100, hyperparameter)  
score = game.main(True)
print("Score: ", score)
