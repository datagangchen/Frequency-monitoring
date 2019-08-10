# this is the main function
from py import *
from binary_tree import *
from fractions import Fraction
import numpy as np
from agenda import *
import random
from struct_formula import *
from robustness import *
import multiprocessing


if __name__ == "__main__":

    state = env.reset()
    prev_x = None
    score = 0
    episode = 0





    state_size = 80 * 80
    action_size = env.action_space.n
    agent = PGAgent(state_size, action_size)
    agent.load('pong.h5')
    while True:
        env.render()

        cur_x = preprocess(state)
        x = cur_x - prev_x if prev_x is not None else np.zeros(state_size)
        prev_x = cur_x

        action, prob = agent.act(x)
        state, reward, done, info = env.step(action)
        score += reward
        agent.remember(x, action, prob, reward)

        if done:
            episode += 1
            agent.train()
            print('Episode: %d - Score: %f.' % (episode, score))
            score = 0
            state = env.reset()
            prev_x = None
            if episode > 1 and episode % 10 == 0:
                agent.save('pong.h5')
