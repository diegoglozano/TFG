#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 19:50:54 2018

@author: diego
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('../data/pacientes_ucic.csv', sep = ';')

minmaxscaler = MinMaxScaler()

data['Tiempo de CEC'] = data['Tiempo de CEC'].fillna(value = 0)
data['Tiempo de clampaje'] = data['Tiempo de clampaje'].fillna(value = 0)
data['Tiempo de isquemia'] = data['Tiempo de isquemia'].fillna(value = 0)
data['Tiempo de parada circulatoria'] = data['Tiempo de parada circulatoria'].fillna(value = 0)

"""
data['Tiempo de CEC'] = minmaxscaler.fit_transform(data[['Tiempo de CEC']])
data['Tiempo de clampaje'] = minmaxscaler.fit_transform(data[['Tiempo de clampaje']])
data['Tiempo de isquemia'] = minmaxscaler.fit_transform(data[['Tiempo de isquemia']])
data['Tiempo de parada circulatoria'] = minmaxscaler.fit_transform(data[['Tiempo de parada circulatoria']])
"""

print(data['Tiempo de CEC'])