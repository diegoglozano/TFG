#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 00:52:55 2018

@author: diego
"""

import pandas as pd

data = pd.read_csv('../data/pacientes_ucic.csv', sep = ';')

"""
0
20.000-49.999
50.000-99.999
100.000-149.999
>150.000
1
"""

data[data['Cifra de plaquetas más baja'] == '20.000-49.999'] = 0
data[data['Cifra de plaquetas más baja'] == '50.000-99.999'] = 1
data[data['Cifra de plaquetas más baja'] == '100.000-149.999'] = 2
data[data['Cifra de plaquetas más baja'] == '>150.000'] = 3

# QUITAMOS LA HORA
data['FECHA_NACIMIENTO'] = data['FECHA_NACIMIENTO'].str.split(" ", 0, True)

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



temporal = pd.get_dummies(data['Ingreso en:'], prefix = 'Ingreso en:', prefix_sep = ' ') #dummy_na = True

data = pd.concat([data, temporal], axis=1, sort=False)


datatemp = pd.get_dummies(data['Motivo de ingreso en UCI'])

data = pd.concat([data, datatemp], axis=1, sort=False)

"""
0
Menor de 3
Mayor o igual a 3
1
"""

data[data['núm. concentrados hematies primeras 48 h'] == 'Menor de 3'] = 0
data[data['núm. concentrados hematies primeras 48 h'] == 'Mayor o igual a 3'] = 1


# HACEMOS UN SPLIT PARA OBTENER LAS CATEGORIAS 
datatemp = data['Otros factores'].str.split(', ')

categorias = []

# RELLENAMOS LOS NAN PARA NO OBTENER ERROR
datatemp = datatemp.fillna(value = 'a')

# RECORREMOS TODAS LAS FILAS Y LOS ELEMENTOS DE CADA FILA Y LOS AÑADIMOS A LA LISTA CATEGORIAS DE FORMA QUE NO SE REPITAN
for i in range(len(datatemp)):
    for j in range(len(datatemp.loc[i])):
        if ((datatemp[i][j] in categorias) == False):
            categorias.append(datatemp[i][j])

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

"""
0
Menor de 500
500-999
1000-3999
4000 o más
1
"""

data[data['Pico de troponina'] == 'Menor de 500'] = 0
data[data['Pico de troponina'] == '500-999'] = 1
data[data['Pico de troponina'] == '1000-3999'] = 2
data[data['Pico de troponina'] == '4000 o más'] = 3


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


data['Tiempo de CEC'] = data['Tiempo de CEC'].fillna(value = 0)
data['Tiempo de clampaje'] = data['Tiempo de clampaje'].fillna(value = 0)
data['Tiempo de isquemia'] = data['Tiempo de isquemia'].fillna(value = 0)
data['Tiempo de parada circulatoria'] = data['Tiempo de parada circulatoria'].fillna(value = 0)


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

cols = data.columns.tolist()
print(data['Cifra de plaquetas más baja'])