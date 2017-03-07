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


import Core as zw

n_calcs = 96
data_fldr = 'C:\Users\mpnun\Desktop\AB_ML'

for i in range(n_calcs):
    
    x = zw.KMC_Run()
    x.Path = os.path.join(data_fldr, str(i))
    x.ReadAllOutput()
    
    print x.Procstat['events'][-1,:] / x.Procstat['t'][-1]