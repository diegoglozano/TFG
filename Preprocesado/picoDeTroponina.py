#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 20:13:44 2018

@author: diego
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer, MinMaxScaler

data = pd.read_csv('../data/pacientes_ucic.csv', sep = ';')

imputer = Imputer(strategy = 'most_frequent')
minmaxscaler = MinMaxScaler()

"""
0
Menor de 500
500-999
1000-3999
4000 o más
1
"""

data[data['Pico de troponina'] == 'Menor de 500'] = 0
data[data['Pico de troponina'] == '500-999'] = 1
data[data['Pico de troponina'] == '1000-3999'] = 2
data[data['Pico de troponina'] == '4000 o más'] = 3

"""
data['Pico de troponina'] = imputer.fit_transform(data[['Pico de troponina']])
data['Pico de troponina'] = minmaxscaler.fit_transform(data[['Pico de troponina']])
"""

print(data['Pico de troponina'])