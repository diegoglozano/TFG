#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 16:49:41 2018

@author: diego
"""

import numpy as np
import pandas as pd
from datetime import datetime

before = datetime.now()

data = pd.read_csv('../data/pacientes_ucic.csv', sep = ';')

#print(data['FECHA_NACIMIENTO'])
#print(data['Fecha ingreso hospitalario'])

"""
# GUARDAMOS EN VARIABLES TEMPORALES DIA MES Y AÑO
temporal0 = data['FECHA_NACIMIENTO'].str.split("/", 0, True)[0]
temporal1 = data['FECHA_NACIMIENTO'].str.split("/", 0, True)[1]
temporal2 = '19' + data['FECHA_NACIMIENTO'].str.split("/", 0, True)[2]

# LAS JUNTAMOS OTRA VEZ
data['FECHA_NACIMIENTO'] = temporal0 + '/' + temporal1 + '/' + temporal2

# CONVERTTIMOS A DATETIME AMBAS COLUMNAS
data['FECHA_NACIMIENTO'] = pd.to_datetime(data['FECHA_NACIMIENTO'])
data['Fecha ingreso hospitalario'] = pd.to_datetime(data['Fecha ingreso hospitalario'])

# CREAMOS LA COLUMNA EDAD Y RESTAMOS LAS FECHAS. LA GUARDAMOS COMO FLOAT Y COMO AÑO 
data['EDAD'] = 0
data['EDAD'] = data['Fecha ingreso hospitalario'] - data['FECHA_NACIMIENTO']
data['EDAD'] = data['EDAD'] / (np.timedelta64(1, 'D') * 365)


# SUSTITUIMOS LOS VALORES 0 (QUE TENIAN UN NAN) POR LA MEDIA CON EL IMPUTER
#data[['EDAD']] = imputer.fit_transform(data[['EDAD']])
#data[['EDAD']] = minmaxscaler.fit_transform(data[['EDAD']])

print(data['EDAD'])

print(datetime.now() - before)
"""

data['Fecha de cirugía'] = pd.to_datetime(data['Fecha de cirugía'])

data['Dias hasta cirugia'] = 0

data['Dias hasta cirugia'] = data['Fecha de cirugía'] - data['Fecha ingreso hospitalario']
data['Dias hasta cirugia'] = data['Dias hasta cirugia'] / (np.timedelta64(1, 'D'))

print(data['Dias hasta cirugia'])