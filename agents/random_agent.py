import numpy as np

class Random_agent:
    def __init__(self):
        pass

    def act(self, state, train):
        prob = np.random.randint(0,12)
        if prob < 1:
            return 1
        else:
            return 0