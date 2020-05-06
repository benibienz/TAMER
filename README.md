# TAMER
TAMER (Training an Agent Manually via Evaluative Reinforcement) is a framework for human-in-the-loop Reinforcement Learning, proposed by [Knox + Stone](http://www.cs.utexas.edu/~sniekum/classes/RLFD-F16/papers/Knox09.pdf) in 2009. 

This is an implementation of a TAMER agent, converted from a standard Q-learning agent using the steps provided by Knox [here](http://www.cs.utexas.edu/users/bradknox/kcap09/Knox_and_Stone,_K-CAP_2009.html).



## How to run
You need python 3.7+ with numpy, sklearn, pygame and gym.

Use run.py. You can fiddle with the config in the script.

In training, watch the agent play and press 'W' to give a positive reward and 'A' to give a negative. The agent's current action is displayed.

![Screenshot of TAMER](screenshot.png)