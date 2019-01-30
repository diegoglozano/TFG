#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 02:12:34 2018

@author: diego
"""

import pandas as pd
import numpy as np

data = pd.read_csv('../data/pacientes_ucic.csv', sep = ';')

temporal = pd.get_dummies(data['SEXO'])

data = pd.concat([data, temporal], axis=1, sort=False)

print(data)