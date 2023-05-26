from Project_5 import *

func_a = lambda x : (x[0] - 4)**2 + x[1]**2
grad_a = lambda x : np.array([2*(x[0]-4), 2*x[1]])

c_a = lambda x : np.array([x[0] + x[1] -2, -x[0], -x[1]])
grad_c_a = lambda x : np.array([[1,1],[-1,0],[0,-1]])