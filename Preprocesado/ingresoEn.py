#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 00:00:40 2018

@author: diego
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, Imputer, MinMaxScaler
from datetime import datetime

data = pd.read_csv('../data/pacientes_ucic.csv', sep = ';')

temporal = pd.get_dummies(data['Ingreso en:'], prefix = 'Ingreso en:', prefix_sep = ' ') #dummy_na = True

data = pd.concat([data, temporal], axis=1, sort=False)

print(list(data))