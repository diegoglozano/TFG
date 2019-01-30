#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 20:24:11 2018

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
Menor de 500 ml
500-1000 ml
Mayor de 1000 ml
1
"""

data[data['Sangrado en las primeras 24 horas'] == 'Menor de 500 ml'] = 0
data[data['Sangrado en las primeras 24 horas'] == '500-1000 ml'] = 1
data[data['Sangrado en las primeras 24 horas'] == 'Mayor de 1000 ml'] = 2

"""
data['Sangrado en las primeras 24 horas'] = imputer.fit_transform(data[['Sangrado en las primeras 24 horas']])
data['Sangrado en las primeras 24 horas'] = minmaxscaler.fit_transform(data[['Sangrado en las primeras 24 horas']])
"""

print(data['Sangrado en las primeras 24 horas'])