# Tiler function for PCapacity under FIRM_CE
# Copyright (c) 2025 Owen Chenhall
# Licensed under the MIT Licence
# Correspondence: owen.chenhall@gmail.com

import numpy as np
from numba import jit

@jit(nopython=True)
def PCapTCalc(CPHP, steps, intervals):
    split = intervals // steps
    Pcapacityt = np.empty((intervals, 1), dtype=np.float64)
    
    for i in range(steps):
        value = np.sum(CPHP[i::steps]) * 1e3
        start = i * split
        end = start + split
        Pcapacityt[start:end, 0] = value

    return Pcapacityt