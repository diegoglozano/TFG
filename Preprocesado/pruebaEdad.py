#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 18:03:52 2018

@author: diego
"""

import pandas as pd

data = pd.read_csv('../data/pacientes_ucic_v3.csv', sep = ';')

print(data['EDAD'])
print(data['edad'])