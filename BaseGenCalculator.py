# A module to populate a BaseGen input array for use within FIRM_CE
# Copyright (c) 2025 Owen Chenhall
# Licensed under the MIT Licence
# Correspondence: owen.chenhall@gmail.com


# Define BaseGen as an array of arrays where each child array is the baseload at each node specified in the order 
# ['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA']. The parent array should be as long as the number of time 
# steps considered. Must be dtype=np.float64
#
# ie. If considering a single node optimisation and the base generation is 50 GW in the first step and reduced to 0 
# in the second then BaseGen = [50,0]

import numpy as np

def tileBaseGen(steps, intervals):
    if (steps == 1):
        BaseGen = np.array([0], dtype=np.float64)
    elif (steps == 2):
        BaseGen = np.array([2750, 0], dtype=np.float64)
    elif (steps == 4):
        BaseGen = np.array([5465, 2750, 1430, 0], dtype=np.float64)   
    elif (steps == 8):
        BaseGen = np.array([8387, 5465, 5465, 2750, 1430, 1430, 1430, 0], dtype=np.float64)  
    else:
        BaseGen = np.array([0]*steps, dtype=np.float64)

    print("base", BaseGen)

    basegen_temp = []
    split = int(intervals/steps)
    for i in range(steps):
        basegen_temp.append(np.tile(BaseGen[i], (split, 1)))

    basegen = np.vstack(basegen_temp)
    return basegen
        

    
