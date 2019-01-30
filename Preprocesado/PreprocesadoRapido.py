#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 17:38:21 2018

@author: diego
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC
import collections
from datetime import datetime
import os
import warnings

warnings.filterwarnings('ignore') # LO AÑADO TEMPORALMENTE PARA NO MOSTRAR LOS WARNINGS DE PANDAS

os.environ['KMP_DUPLICATE_LIB_OK']='True' # LO AÑADO TEMPORALMENTE PORQUE XGBOOST DABA UN ERROR CON MACOS

before = datetime.now()

print('PREPROCESANDO DATOS\n')

# LEEMOS DATOS
df = pd.read_csv('../data/pacientes_ucic_v3.csv', sep = ';')

# SEPARAMOS EN X E Y
X = df.iloc[:, 1:221] # LA 0 NO LA QUIERO. DE LA 1 A LA 8 SON CARDINALES. EL RESTO ORDINALES
y = df['Situación al alta de UCI.Control de fallo cardiaco'] # TIENE UNA PROPORCION DE 306/104 (75%-25%)

# SEPARAMOS EN TRAIN Y TEST
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 11) # (307, 220) (103, 220) (307,) (103,) POR DEFECTO 75/25

X_train_Media = X_train.iloc[:, 0:8]
X_test_Media = X_test.iloc[:, 0:8]
X_train_Moda = X_train.iloc[:, 8:]
X_test_Moda = X_test.iloc[:, 8:]

# PREPROCESADO
imputerMedia = SimpleImputer(strategy = 'mean') # VARIABLES CARDINALES
imputerModa = SimpleImputer(strategy = 'most_frequent') # VARIABLES ORDINALES
standardMedia = StandardScaler()
standardModa = StandardScaler()

imputerMedia.fit(X_train_Media)
imputerModa.fit(X_train_Moda)
standardMedia.fit(X_train_Media)
standardModa.fit(X_train_Moda)

print('Tiempo tras declarar y fit: {}\n' .format(datetime.now() - before))

# VARIABLES CARDINALES (MEDIA)
X_train_Media = imputerMedia.transform(X_train_Media)
print('Tiempo 1: {}\n' .format(datetime.now() - before))
X_train_Media = standardMedia.transform(X_train_Media)
print('Tiempo 2: {}\n' .format(datetime.now() - before))
X_test_Media = imputerMedia.transform(X_test_Media)
print('Tiempo 3: {}\n' .format(datetime.now() - before))
X_test_Media = standardMedia.transform(X_test_Media)
print('Tiempo 4: {}\n' .format(datetime.now() - before))

# VARIABLES ORDINALES (MODA)
X_train_Moda = imputerModa.transform(X_train_Moda) # TARDA 8 SEGUNDOS
print('Tiempo 5: {}\n' .format(datetime.now() - before))
X_train_Moda = standardModa.transform(X_train_Moda)
print('Tiempo 6: {}\n' .format(datetime.now() - before))
X_test_Moda = imputerModa.transform(X_test_Moda) # TARDA 8 SEGUNDOS
print('Tiempo 7: {}\n' .format(datetime.now() - before))
X_test_Moda = standardModa.transform(X_test_Moda)
print('Tiempo 8: {}\n' .format(datetime.now() - before))

X_train_np = np.concatenate([X_train_Media, X_train_Moda], axis = 1)
X_test_np = np.concatenate([X_test_Media, X_test_Moda], axis = 1)

X_train = pd.DataFrame(X_train_np, columns = X_train.columns)
X_test = pd.DataFrame(X_test_np, columns = X_test.columns)

print(X_train)