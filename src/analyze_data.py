# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 15:30:35 2017

@author: mpnun
"""

import sys
import os
import numpy as np
import matplotlib as mat
import matplotlib.pyplot as plt
import copy
import random
from shutil import copyfile
import xlsxwriter
import pickle

import Core as zw

n_calcs = 100
data_fldr = 'C:\Users\mpnun\Desktop\data2'

freq_data = [[] for i in range(n_calcs)]
rate_data = [[] for i in range(n_calcs)]

for i in range(n_calcs):
    
    x = zw.KMC_Run()
    x.Path = os.path.join(data_fldr, str(i))
    x.ReadAllOutput()
    
    step_freqs = x.Procstat['events'][-1,:] / x.Procstat['t'][-1]
    rate = x.Specnum['spec'][-1,-2] / x.Specnum['t'][-1]
    
    freq_data[i] = step_freqs
    rate_data[i] = rate
    
freq_data = np.array(freq_data)
rate_data = np.array(rate_data)

with open('freq.pickle','w') as f:
    pickle.dump(freq_data, f)
    
with open('rate.pickle','w') as f:
    pickle.dump(rate_data, f)
    
print freq_data
print rate_data