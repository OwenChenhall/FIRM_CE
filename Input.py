# Modelling input and assumptions
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Modifications (c) 2025 Owen Chenhall
# Licensed under the MIT License
# Original author: Bin Lu (bin.lu@anu.edu.au)
# Modified by: Owen Chenhall

# Discription of changes (2025, Owen Chenhall)
# - Added support capacity expansion under FIRM_CE. Key Changes to solution class sturcture

import numpy as np
from Optimisation import scenario, node, steps
from numba import float64, int32, types, int64
from numba.experimental import jitclass
import BaseGenCalculator

# Build limits for capacity in each timestep. Sets upperbound of optimisation
PVBuildRateLimit = 100    #GW/step
WindBuildRateLimit = 100  #GW/step

DemandGrowth = 1          #Load multiplier per step


Nodel = np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA']) #Defines nodes in scenario
PVl =   np.array(['NSW']*7 + ['FNQ']*1 + ['QLD']*2 + ['FNQ']*3 + ['SA']*6 + ['TAS']*0 + ['VIC']*1 + ['WA']*1 + ['NT']*1) #Defines PV sites in scenario. Match to header of PV data input 
Windl = np.array(['NSW']*8 + ['FNQ']*1 + ['QLD']*2 + ['FNQ']*2 + ['SA']*8 + ['TAS']*4 + ['VIC']*4 + ['WA']*3 + ['NT']*1) #Defines Wind sites in scenario. Match to header of Wind data input

_, Nodel_int = np.unique(Nodel, return_inverse=True)
_, PVl_int = np.unique(Nodel, return_inverse=True)
_, Windl_int = np.unique(Nodel, return_inverse=True)

Nodel_int = Nodel_int.astype(np.int32)
PVl_int = PVl_int.astype(np.int32)
Windl_int = Windl_int.astype(np.int32)


# Time intervals. The time between data points of input data in hours
resolution = 0.5 


#Data import handling
MLoad = np.genfromtxt('Data/electricity16year.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(Nodel))) # EOLoad(t, j), MW
TSPV = np.genfromtxt('Data/pv16year.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(PVl))) # TSPV(t, i), MW
TSWind = np.genfromtxt('Data/wind16year.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(Windl))) # TSWind(t, i), MW
assets = np.genfromtxt('Data/hydrobio.csv', dtype=None, delimiter=',', encoding=None)[1:, 1:].astype(np.float64)
basegen = np.genfromtxt('Data/baseload.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(Nodel)))
CHydro, CBio = [assets[:, x] * pow(10, -3) for x in range(assets.shape[1])] # CHydro(j), MW to GW
CBaseload = np.array([0, 0, 0, 0, 0, 0, 0, 0]) # 24/7, GW
CPeak = CHydro + CBio - CBaseload # GW


#Defines transmission lines considered in network
# FQ, NQ, NS, NV, AS, SW, only TV constrained
DCloss = np.array([1500, 1000, 1000, 800, 1200, 2400, 400]) * 0.03 * pow(10, -3)

efficiency = 0.8
factor = np.genfromtxt('Data/factor.csv', delimiter=',', usecols=1)


if node == 'Super1': #Defult option
    coverage = Nodel
else:
    coverage = np.array([node]) # For single node optimisations

# Adjust data inports to match scenario (changes which nodes are considered)
MLoad = MLoad[:, np.where(np.in1d(Nodel, coverage)==True)[0]]
TSPV = TSPV[:, np.where(np.in1d(PVl, coverage)==True)[0]]
TSWind = TSWind[:, np.where(np.in1d(Windl, coverage)==True)[0]]
basegen = basegen[:, np.where(np.in1d(Nodel, coverage)==True)[0]]

CHydro, CBio, CBaseload, CPeak = [x[np.where(np.in1d(Nodel, coverage)==True)[0]] for x in (CHydro, CBio, CBaseload, CPeak)]

Nodel_int, PVl_int, Windl_int = [x[np.where(Nodel==node)[0]] for x in (Nodel_int, PVl_int, Windl_int)]
Nodel, PVl, Windl = [x[np.where(x==node)[0]] for x in (Nodel, PVl, Windl)]


# Apply load multiplier
MLoad_split = int(len(MLoad)/steps)
for i in range(steps):
    if i == 0:
        MLoad[MLoad_split*i:MLoad_split*(i+1)] = MLoad[MLoad_split*i:MLoad_split*(i+1)]
    else:
        MLoad[MLoad_split*i:MLoad_split*(i+1)] = MLoad[MLoad_split*i:MLoad_split*(i+1)] * (DemandGrowth ** i)
    

intervals, nodes = MLoad.shape
years = int(resolution * intervals / 8760)
pzones, wzones = (TSPV.shape[1] * steps, TSWind.shape[1] * steps)


pidx, widx, sidx = (pzones, pzones + wzones, pzones + wzones + (nodes * steps))


energy = MLoad.sum() * pow(10, -9) * resolution / years # PWh p.a.
contingency = list(0.25 * MLoad.max(axis=0) * pow(10, -3)) # MW to GW

GBaseload = BaseGenCalculator.tileBaseGen(steps, intervals)
#GBaseload = np.tile(CBaseload, (intervals, 1)) * pow(10, 3) # GW to MW # Used if importing base generation from CSV


# Specify the types for jitclass
solution_spec = [
    ('steps', float64),
    ('x', float64[:]),  # Assuming x is a list of floats
    ('MLoad', float64[:, :]),  # 2D array of floats
    ('intervals', int32),
    ('nodes', int32),
    ('resolution',float64),
    ('CPV', float64[:]),
    ('CWind', float64[:]),
    ('GPV', float64[:, :]),  # 2D array of floats
    ('GWind', float64[:, :]),  # 2D array of floats
    ('CPHP', float64[:]),
    ('CPHS', float64),
    ('efficiency', float64),
    ('CInter', float64[:]),
    ('GInter', float64[:, :]),  # 2D array of floats
    ('Nodel_int', int32[:]), 
    ('PVl_int', int32[:]),
    ('Windl_int', int32[:]),
    ('Interl_int', int32[:]),
    ('node', types.unicode_type),
    ('GBaseload', float64[:, :]),  # 2D array of floats
    ('CPeak', float64[:]),  # 1D array of floats
    ('CHydro', float64[:]),  # 1D array of floats
    ('EHydro', float64[:]),  # 1D array of floats
    ('allowance', float64),
    ('flexible', float64[:,:]),
    ('Discharge', float64[:,:]),
    ('Charge', float64[:,:]),
    ('Storage', float64[:,:]),
    ('Deficit', float64[:,:]),
    ('Spillage', float64[:,:])
]


@jitclass(solution_spec)
class Solution:
    #A candidate solution of decision variables (CPV(i), CWind(j), CPHP(k)) * steps, S-CPHS(l)
    
    def __init__(self, x):
        self.steps = steps
        self.x = x
        self.MLoad = MLoad
        self.intervals = intervals
        self.nodes = nodes
        self.resolution = resolution


        self.CPV = self.x[: pidx].copy()  # CPV(i), GW
        self.CWind = self.x[pidx : widx].copy()  # CWind(j), GW
        self.CPHP = self.x[widx : sidx].copy()  # CPHP(k), GW
        self.CPHS = self.x[sidx]  # S-CPHS(l), GWh
    
   
        # Modify CPV, CWind, and CPHP to represent cumulative capacities
        CPV_split = int(len(self.CPV)/steps)
        for i in range(steps - 1):
            i = i + 1 #dont apply to the base case
            self.CPV[CPV_split*i:CPV_split*(i+1)] = self.CPV[CPV_split*i:CPV_split*(i+1)] + self.CPV[CPV_split*(i-1):CPV_split*(i)]
        
        CWind_split = int(len(self.CWind)/steps)
        for i in range(steps - 1):
            i = i + 1 #dont apply to the base case
            self.CWind[CWind_split*i:CWind_split*(i+1)] = self.CWind[CWind_split*i:CWind_split*(i+1)] + self.CWind[CWind_split*(i-1):CWind_split*(i)]

        CPHP_split = int(len(self.CPHP)/steps)
        for i in range(steps - 1):
            i = i + 1 #dont apply to the base case
            self.CPHP[CPHP_split*i:CPHP_split*(i+1)] = self.CPHP[CPHP_split*i:CPHP_split*(i+1)] + self.CPHP[CPHP_split*(i-1):CPHP_split*(i)]


        # Manually replicating np.tile functionality for CPV and CWind
        CPV_tiled = np.zeros((intervals, CPV_split))
        CWind_tiled = np.zeros((intervals, CWind_split))
        
        intervals_per_step = intervals/steps
        for i in range(intervals):
            step_idx = i // intervals_per_step  # which segment to use
            for j in range(CPV_split):
                CPV_tiled[i, j] = self.CPV[int(j + (step_idx * CPV_split))]
            for j in range(CWind_split):
                CWind_tiled[i, j] = self.CWind[int(j + (step_idx * CWind_split))]
            """ for j in range(len(self.CInter)):
                CInter_tiled[i, j] = self.CInter[j] """


        self.GPV = TSPV * CPV_tiled * 1000  # GPV(i, t), GW to MW
        self.GWind = TSWind * CWind_tiled * 1000  # GWind(i, t), GW to MW
        
        self.efficiency = efficiency

        self.Nodel_int = Nodel_int
        self.PVl_int = PVl_int
        self.Windl_int = Windl_int
        self.node = node

        self.GBaseload = GBaseload
        self.CPeak = CPeak
        self.CHydro = CHydro
