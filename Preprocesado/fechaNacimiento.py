#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 16:36:16 2018

@author: diego
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, Imputer, MinMaxScaler
from datetime import datetime


data = pd.read_csv('../data/pacientes_ucic.csv', sep = ';')

#print(data['FECHA_NACIMIENTO'])
#print(data['Fecha ingreso hospitalario'])

# QUITAMOS LA HORA
for i in range(len(data['FECHA_NACIMIENTO'])):
    data['FECHA_NACIMIENTO'].loc[i] = str(data['FECHA_NACIMIENTO'].loc[i]).split()[0]

# LO SUSTITUYO POR 0/0/0 PARA QUE EL SPLIT FUNCIONE
for i in range(len(data['Fecha ingreso hospitalario'])):
    if(pd.isnull(data['Fecha ingreso hospitalario'].loc[i])):
                data['Fecha ingreso hospitalario'].loc[i] = '1/1/0'

# DIVIDIMOS LA STRING, LE SUMAMOS 1900 O 2000 PARA QUE TENGA EL FORMATO CORRECTO Y LO VOLVEMOS A UNIR
for i in range(len(data['FECHA_NACIMIENTO'])):
    temporalNacimiento = str(data['FECHA_NACIMIENTO'].loc[i]).split('/')
    temporalIngreso = str(data['Fecha ingreso hospitalario'].loc[i]).split('/')
    temporalNacimiento[2] = int(temporalNacimiento[2]) + 1900
    temporalIngreso[2] = int(temporalIngreso[2]) + 2000
    for j in range(3):
        temporalNacimiento[j] = str(temporalNacimiento[j])
        temporalIngreso[j] = str(temporalIngreso[j])
    temporalNacimiento = '/'.join(temporalNacimiento)
    temporalIngreso = '/'.join(temporalIngreso)
    data['FECHA_NACIMIENTO'].loc[i] = temporalNacimiento
    data['Fecha ingreso hospitalario'].loc[i] = temporalIngreso

#print(data['FECHA_NACIMIENTO'])
#print(data['Fecha ingreso hospitalario'])

formato_fecha = "%d/%m/%Y"

data['EDAD'] = 0

for i in range(len(data['FECHA_NACIMIENTO'])):
    fecha_inicial = datetime.strptime(data['FECHA_NACIMIENTO'].loc[i], formato_fecha)
    fecha_final = datetime.strptime(data['Fecha ingreso hospitalario'].loc[i], formato_fecha)
    data['EDAD'].loc[i] = (fecha_final - fecha_inicial).days/365

print(data['EDAD'])