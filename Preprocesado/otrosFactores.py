#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 01:09:58 2018

@author: diego
"""

import pandas as pd
import numpy as np
from datetime import datetime

before = datetime.now()

data = pd.read_csv('../data/pacientes_ucic.csv', sep = ';')

# HACEMOS UN SPLIT PARA OBTENER LAS CATEGORIAS 
datatemp = data['Otros factores'].str.split(', ')

print('1:', datetime.now() - before)

categorias = []

# RELLENAMOS LOS NAN PARA NO OBTENER ERROR
datatemp = datatemp.fillna(value = 'a')

# RECORREMOS TODAS LAS FILAS Y LOS ELEMENTOS DE CADA FILA Y LOS AÑADIMOS A LA LISTA CATEGORIAS DE FORMA QUE NO SE REPITAN
for i in range(len(datatemp)):
    for j in range(len(datatemp.loc[i])):
        if ((datatemp[i][j] in categorias) == False):
            categorias.append(datatemp[i][j])

print('2:', datetime.now() - before)

# QUITAMOS LA CATEGORÍA 'A' (NAN)
categorias.remove('a')

# RELLENO TODO CON FALSE
for i in categorias:
    data['Otros factores. ' + i] = 0

for i in range(len(datatemp)): # RECORRO FILAS
    for j in range(len(datatemp.loc[i])): # RECORRO ELEMENTOS DENTRO DE UNA FILA
        for k in range(len(categorias)): # RECORRO CATEGORIAS
            if(datatemp[i][j] == categorias[k]):
                data['Otros factores. ' + categorias[k]].loc[i] = 1
            """
            else:
                data['Otros factores. ' + categorias[k]].loc[i] = False
            """
# CREO QUE EL PROBLEMA ES QUE SE ESTA SOBREESCRIBIENDO
# POR EJEMPLO: EN LA PRIMERA ITERACION LA PRIMERA FILA DE APERTURA PLEURAS SE PONE EN TRUE PERO AL COMPARAR CON LA SIGUIENTE
# CATEGORIA SE SOBREESCRIBE POR UN FALSE DADO QUE EL BUCLE FOR DE LAS FILAS TODAVIA NO CAMBIO. HAY QUE HACER UN BREAK

# LA SOLUCION PUEDE SER RELLENAR TODO CON FALSE Y DESPUES CAMBIAR A TRUE SI SE DA EL CASO. FUE BUENA LA SOLUCION

# HAY QUE CAMBIAR TRUE Y FALSE POR 1 Y 0

print('3:', datetime.now() - before)

print(data['Otros factores. Apertura pleuras'])

print('Total:', datetime.now() - before)