#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 20:34:12 2018

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
Menor de 3
Mayor o igual a 3
1
"""

data[data['núm. concentrados hematies primeras 48 h'] == 'Menor de 3'] = 0
data[data['núm. concentrados hematies primeras 48 h'] == 'Mayor o igual a 3'] = 1

"""
data['núm. concentrados hematies primeras 48 h'] = imputer.fit_transform(data[['núm. concentrados hematies primeras 48 h']])
data['núm. concentrados hematies primeras 48 h'] = minmaxscaler.fit_transform(data[['núm. concentrados hematies primeras 48 h']])
"""

print(data['núm. concentrados hematies primeras 48 h'])