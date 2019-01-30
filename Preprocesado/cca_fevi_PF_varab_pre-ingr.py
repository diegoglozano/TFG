#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 17:29:25 2018

@author: diego
"""

import pandas as pd

data = pd.read_csv('../data/pacientes_ucic.csv', sep = ';')

data['cca_fevi_PF_varab_pre-ingr'] = data['cca_fevi_PF_varab_pre-ingr'].fillna(value = 'No cuantificada')
cat = pd.Categorical(data['cca_fevi_PF_varab_pre-ingr'], categories=['No cuantificada', 'Normal', 'Disfunción leve (>40)', 'Disfunción moderada (30-40)', 'Disfunción grave (<30)'])
data['cca_fevi_PF_varab_pre-ingr'], uniques = pd.factorize(cat, sort = True)

print(data['cca_fevi_PF_varab_pre-ingr'])