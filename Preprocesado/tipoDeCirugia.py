#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 00:41:47 2018

@author: diego
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer, MinMaxScaler
from datetime import datetime

before = datetime.now()

data = pd.read_csv('../data/pacientes_ucic.csv', sep=';')

imputer = Imputer()
minmaxscaler = MinMaxScaler()

"""
0 Peor
NaN
Programada
Emergente
Urgente
1 Mejor
"""

data['Tipo de cirugía'] = data['Tipo de cirugía'].fillna(value = 0)
data[data['Tipo de cirugía'] == 'Programada'] = 1
data[data['Tipo de cirugía'] == 'Emergente (intervención en menos de 48 horas del diagnóstico'] = 2
data[data['Tipo de cirugía'] == 'Urgente (se interviene durante el ingreso de una descompensa'] = 3

"""
data['Tipo de cirugía'] = minmaxscaler.fit_transform(data[['Tipo de cirugía']])
"""

print(data['Tipo de cirugía'])

print(datetime.now() - before)