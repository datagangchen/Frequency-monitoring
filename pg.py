#!/usr/local/bin/python3
import numpy as np
import math
from struct_formula import *
from copy import deepcopy
class PGAgent:
    def __init__(self, step =3, episode = 100, name =None, signalset = None, time =None, width=3):
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.states = []
        self.actions = []
        self.rewards = []
        self.probs =[]
        self.formula =[]
        self.step = step
        self.name = name
        self.signal = signalset 
        self.time = time 
        self.width = width 
        self.episode = episode
        self.W = [[np.random.rand(10*width)] for _ in range(step)]
        _, formula = Init_state(self.name,self.signal,self.time, self.width)
        self.formula = formula 


    def predict(self, state, t):
        w = self.W[t]
        return [math.exp(np.dot(w,np.squeeze(np.asarray(vector)))) for vector in state]

    def remember(self, state, action, prob, reward):
        self.actions.append(action)
        self.states.append([state])
        self.rewards.append(reward)
        self.probs.append(prob)


    def get_reward(self, tree):
        return np.min(reward(tree, self.name, self.signal, self.time))

    def get_poolreward(self,tree ):
        return np.min(poolreward(tree,self.name, self.signal, self.time))

    def run(self, agenda,chart):
        for t in range(self.step):
            state = self.formula.agenda_vector(agenda)
            action, prob = self.act(state,t)
            tree = self.formula.vector_tree(action[0].tolist())
            reward = self.get_reward(tree)
            self.formula.update_agenda(agenda,chart, tree)
            self.remember(state,action,prob,reward)
        dis_rewards = self.discount_rewards(self.rewards)
        for t in range(self.step):
            gradient = self.actions[t] - np.sum([a*b[0] for a, b in zip(self.probs[t], self.states[t][0])])
            self.W[t] += self.learning_rate*dis_rewards[t]*gradient

        
        


    def act(self, state,t):
        aprob = self.predict(state,t)
        prob = aprob / np.sum(aprob)
        action = np.random.choice(len(state), 1, p=prob)[0]
        return state[action], prob

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def train(self):
        agenda0, chart0 =[],[]
        self.formula.init_agenda(agenda0, chart0, 50)
        for index in range(self.episode):
            agenda, chart = deepcopy(agenda0), deepcopy(chart0)
            self.run(agenda, chart)
            if np.max(self.rewards)>0:
                break
            if index == self.episode-1:
                return
            else:
                self.probs, self.states, self.rewards, self.actions = [],[],[],[]

        



        

        








