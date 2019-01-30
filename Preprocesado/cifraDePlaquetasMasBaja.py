#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 20:27:30 2018

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
20.000-49.999
50.000-99.999
100.000-149.999
>150.000
1
"""

data[data['Cifra de plaquetas más baja'] == '20.000-49.999'] = 0
data[data['Cifra de plaquetas más baja'] == '50.000-99.999'] = 1
data[data['Cifra de plaquetas más baja'] == '100.000-149.999'] = 2
data[data['Cifra de plaquetas más baja'] == '>150.000'] = 3

"""
data['Cifra de plaquetas más baja'] = imputer.fit_transform(data[['Cifra de plaquetas más baja']])
data['Cifra de plaquetas más baja'] = minmaxscaler.fit_transform(data[['Cifra de plaquetas más baja']])
"""

print(data['Cifra de plaquetas más baja'])