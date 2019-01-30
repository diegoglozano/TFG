#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 00:08:06 2018

@author: diego
"""

import pandas as pd
from sklearn.preprocessing import Imputer, MinMaxScaler
from datetime import datetime

before = datetime.now()

data = pd.read_csv('../data/pacientes_ucic.csv', sep=';')

minmaxscaler = MinMaxScaler()

"""
0 Mejor?
nan
Muy saludable. Por encima de lo esperado para su edad
Sano: Paciente asintomático, con actividad normal
Controlado: Paciente con sintomatología tratada, con activid
Vulnerable:Independiente, pero con actividad limitada por su
Fragilidad leve: Necesita ayuda para tareas difíciles
Fragilidad moderada: Necesita ayuda dentro y fuera de la cas
Fragilidad grave: Totalmente dependiente para las actividade
Fragilidad muy grave: Totalmente dependiente, no tolera ni l
Enfermedad terminal: Esperanza de vida menor de 6 meses
1 Peor?
"""
        
data['Índice de Fragilidad'] = data['Índice de Fragilidad'].fillna(value = 0)
data[data['Índice de Fragilidad'] == 'Muy saludable. Por encima de lo esperado para su edad'] = 1
data[data['Índice de Fragilidad'] == 'Sano: Paciente asintomático, con actividad normal'] = 2
data[data['Índice de Fragilidad'] == 'Controlado: Paciente con sintomatología tratada, con activid'] = 3
data[data['Índice de Fragilidad'] == 'Vulnerable:Independiente, pero con actividad limitada por su'] = 4
data[data['Índice de Fragilidad'] == 'Fragilidad leve: Necesita ayuda para tareas difíciles'] = 5
data[data['Índice de Fragilidad'] == 'Fragilidad moderada: Necesita ayuda dentro y fuera de la cas'] = 6
data[data['Índice de Fragilidad'] == 'Fragilidad grave: Totalmente dependiente para las actividade'] = 7
data[data['Índice de Fragilidad'] == 'Fragilidad muy grave: Totalmente dependiente, no tolera ni l'] = 8
data[data['Índice de Fragilidad'] == 'Enfermedad terminal: Esperanza de vida menor de 6 meses'] = 9

"""
data['Índice de Fragilidad'] = minmaxscaler.fit_transform(data[['Índice de Fragilidad']])
"""

print(data['Índice de Fragilidad'])

print(datetime.now() - before)