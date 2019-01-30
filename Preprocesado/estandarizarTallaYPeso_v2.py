#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 01:40:28 2018

@author: diego
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, Imputer, MinMaxScaler


data = pd.read_csv('../data/pacientes_ucic.csv', sep = ';')

#Seguro?
scaler = StandardScaler()
minmaxscaler = MinMaxScaler()
imputer = Imputer(missing_values = 'NaN', strategy = 'mean')

data[['Talla']] = imputer.fit_transform(data[['Talla']])
data[['Peso']] = imputer.fit_transform(data[['Peso']])

data[['Talla']] = minmaxscaler.fit_transform(data[['Talla']])
data[['Peso']] = minmaxscaler.fit_transform(data[['Peso']])

print(data['Talla'])
print(data['Peso'])