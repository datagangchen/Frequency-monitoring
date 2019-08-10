from binary_tree import *
from fractions import Fraction
import numpy as np
import random
from struct_formula import *
from robustness import *
import multiprocessing

class Agenda:
    def __init__(self, low_time =0.0, up_time = 1.0, low_pre =0.0, up_pre=1.0, width = 100):
        self.chart = []
        self.agenda =[]

    def updata_agenda(self, action):
        self.chart = self.chart.append(action)

    def get_agenda_state(self):
        self.get_agenda_state

    def get_reward(self, action):
    


