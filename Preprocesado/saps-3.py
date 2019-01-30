#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 20:07:04 2018

@author: diego
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer, MinMaxScaler

data = pd.read_csv('../data/pacientes_ucic.csv', sep = ';')

imputer = Imputer()
minmaxscaler = MinMaxScaler()

data['SAPS-3'] = imputer.fit_transform(data[['SAPS-3']])

"""
data['SAPS-3'] = minmaxscaler.fit_transform(data[['SAPS-3']])
"""

print(data['SAPS-3'])