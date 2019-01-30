#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 14:13:40 2018

@author: diego
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, Imputer, MinMaxScaler


data = pd.read_csv('../data/pacientes_ucic.csv', sep = ';')

#Seguro?
scaler = StandardScaler()
minmaxscaler = MinMaxScaler()
imputer = Imputer(missing_values = 0, strategy = 'mean')

#data['Talla'] = scaler.fit([data['Talla']])
#data['Talla'] = scaler.transform(data['Talla'])

# LO SUSTITUYO POR 0 PARA QUE EL DTYPE SEA UN FLOAT64
for i in range(len(data['Talla'])):
    if(pd.isnull(data['Talla'].loc[i])):
                data['Talla'].loc[i] = 0

for i in range(len(data['Peso'])):
    if(pd.isnull(data['Peso'].loc[i])):
                data['Peso'].loc[i] = 0     

data[['Talla']] = imputer.fit_transform(data[['Talla']])
data[['Peso']] = imputer.fit_transform(data[['Peso']])

data[['Talla']] = minmaxscaler.fit_transform(data[['Talla']])
data[['Peso']] = minmaxscaler.fit_transform(data[['Peso']])

print(data['Talla'])
print(data['Peso'])