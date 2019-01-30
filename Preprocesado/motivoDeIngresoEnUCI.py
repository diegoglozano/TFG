#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 20:02:15 2018

@author: diego
"""

import pandas as pd
import numpy as np

data = pd.read_csv('../data/pacientes_ucic.csv', sep = ';')

datatemp = pd.get_dummies(data['Motivo de ingreso en UCI'], prefix = 'Motivo de ingreso en UCI:', prefix_sep = ' ')

data = pd.concat([data, datatemp], axis=1, sort=False)

print(list(data))
print(data['Motivo de ingreso en UCI: Post-operatorio C.Cardiaca'])