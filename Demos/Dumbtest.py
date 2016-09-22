"""
Created by Dan on 09/21/16, dumbtest of the methods on the laptop 
"""
import sys
sys.path.insert(0, '../src')
import matplotlib.pyplot as plt
import time
from Ray_trace import Trace
from microscope import objective


def dumb1():
    """
    dumb test 
    """
    Olympus = objective(NA=1.0, ref_ind = 1.33, FL = 9.0, wd = 2.0)
    Rtr = Trace(Olympus)
    
    
    
if __name__ == '__main__':
    start_time = time.time()
    dumb1()

    print("--- %s seconds ---" % (time.time() - start_time)) 