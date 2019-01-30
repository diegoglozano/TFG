#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 18:29:42 2018

@author: diego
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, Imputer, MinMaxScaler
from datetime import datetime

before = datetime.now()

data = pd.read_csv('../data/pacientes_ucic.csv', sep = ';')

imputer = Imputer(missing_values = 0, strategy = 'mean')
minmaxscaler = MinMaxScaler()

#print(data['FECHA_NACIMIENTO'])
#print(data['Fecha ingreso hospitalario'])

# QUITAMOS LA HORA
for i in range(len(data['FECHA_NACIMIENTO'])):
    data['FECHA_NACIMIENTO'].loc[i] = str(data['FECHA_NACIMIENTO'].loc[i]).split()[0]


# DIVIDIMOS LA STRING, LE SUMAMOS 1900 O 2000 PARA QUE TENGA EL FORMATO CORRECTO Y LO VOLVEMOS A UNIR (EXCEPTO NaN)
for i in range(len(data['FECHA_NACIMIENTO'])):
    if(pd.isnull(data['FECHA_NACIMIENTO'].loc[i]) == False):
        temporalNacimiento = str(data['FECHA_NACIMIENTO'].loc[i]).split('/')
        temporalNacimiento[2] = int(temporalNacimiento[2]) + 1900
        for j in range(3):
            temporalNacimiento[j] = str(temporalNacimiento[j])
        temporalNacimiento = '/'.join(temporalNacimiento)
        data['FECHA_NACIMIENTO'].loc[i] = temporalNacimiento
    if(pd.isnull(data['Fecha ingreso hospitalario'].loc[i]) == False):
        temporalIngreso = str(data['Fecha ingreso hospitalario'].loc[i]).split('/')
        temporalIngreso[2] = int(temporalIngreso[2]) + 2000
        for k in range(3):
            temporalIngreso[k] = str(temporalIngreso[k])
        temporalIngreso = '/'.join(temporalIngreso)
        data['Fecha ingreso hospitalario'].loc[i] = temporalIngreso

#print(data['FECHA_NACIMIENTO'])
#print(data['Fecha ingreso hospitalario'])

formato_fecha = "%d/%m/%Y"

data['EDAD'] = 0

# SI NINGUN VALOR ES NaN RESTAMOS Y SACAMOS EL NUMERO DE AÑOS
for i in range(len(data['FECHA_NACIMIENTO'])):
    if(pd.isnull(data['FECHA_NACIMIENTO'].loc[i]) == False and pd.isnull(data['Fecha ingreso hospitalario'].loc[i]) == False):
        fecha_inicial = datetime.strptime(data['FECHA_NACIMIENTO'].loc[i], formato_fecha)
        fecha_final = datetime.strptime(data['Fecha ingreso hospitalario'].loc[i], formato_fecha)
        data['EDAD'].loc[i] = (fecha_final - fecha_inicial).days/365
    

#print(data['EDAD'])
        
# SUSTITUIMOS LOS VALORES 0 (QUE TENIAN UN NAN) POR LA MEDIA CON EL IMPUTER
data[['EDAD']] = imputer.fit_transform(data[['EDAD']])
data[['EDAD']] = minmaxscaler.fit_transform(data[['EDAD']])

print(data['EDAD'])

print(datetime.now() - before)