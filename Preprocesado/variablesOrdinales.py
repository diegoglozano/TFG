#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 11:06:32 2018

@author: diego
"""
import numpy as np
import pandas as pd

datos = pd.DataFrame([2,3,2,'NA','NA','NA',5,3],columns=['num'])

datos[datos['num']=='NA'] = 0

print(datos)