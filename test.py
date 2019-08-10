#!/usr/local/bin/python3

from robustness import *
from struct_formula import *
from load_data import *
import numpy as np
import time 
from pg import *
# import numpy as np


def runn_program():
	name = ['anadataiot1.mat', 'anadataiot2.mat','anadataiot3.mat','anadataiot4.mat','anadataiot5.mat','anadataiot6.mat']
	sigsets, time1, namelist, label = load_data(name[0])
	Agent = PGAgent(2, 200, namelist, sigsets, time1, width = 3)
	t1 = time()
	Agent.train()
	t2 = time()
	reward = Agent.rewards
	actions = Agent.actions
	#formula = Agent.formula
	#formula.vector_tree(actions[index])
	index = reward.index(max(reward))
	print(reward[index])
	print(actions[index])
	print(t2-t1)





	




	

    

	















	#time = np.linspace(0,10,1001)
	#signal = np.array([time])
	#name = ['x1']
	#system = STL_Sys(name,signal,time)
	# Calculate robustness
	#Robust.Eval_Robust(system)
	

if __name__ == '__main__':
	runn_program()
