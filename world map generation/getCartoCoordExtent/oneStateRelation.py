"""
Find out the states with the specified relation with one state
The relations include adjacency, distance and orientation relation
"""

from osgeo import ogr
import numpy as np
import pickle
import sys
sys.path.append('C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\spatial information analysis')
from shapex import *

def adjStates(state, stateAdjMat, stateNames):
    idxState = findIndex(state,stateNames)
    adjStateList = []
    lenStates = len(stateAdjMat[idxState])
    for i in range(lenStates):
        if stateAdjMat[idxState][i] == 1:
            stateName = stateNames[i]
            adjStateList.append(stateName)
    return adjStateList

def findIndex(stateName, stateNames):
    if not (stateName in stateNames):
        print("state name not found")
        return
    return stateNames.index(stateName)

if __name__ == "__main__":
    # two states
    state = 'Ohio'
    # load state adjacency array to be used for adjacency relation identification
    adjFile = 'C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\spatial information analysis\\USStateAdj.npy'
    stateAdjMat = np.load(adjFile)

    with open('C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\spatial information analysis\\stateNames.pickle', 'rb') as f:
        stateNames = pickle.load(f)
    # identify adjacency relation
    adjStateList = adjStates(state, stateAdjMat, stateNames)

    print(adjStateList)