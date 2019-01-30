import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, cross_validate
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


PATH = r'../../Data/pacientes_ucic_v3.csv'
random_state = 11
kernel = ['linear']
C = np.arange(-4.0, 2.0)
C = 10**C
param_grid = [{'svm__C': C, 'svm__kernel': kernel}]  # Hay que acceder a los parametros del pipeline como estimator__parameter

df = pd.read_csv(PATH, sep=';', index_col='Unnamed: 0')

cols = df.columns

y_cols = []
for col in cols:
    y_cols.append(col) if 'Situación al alta' in col else None

X_cols = df.drop(y_cols, axis=1).columns
y_col = 'Situación al alta de UCI.Precisa telemetría'

X = df[X_cols]
y = df[y_col]

imputer_media = SimpleImputer(strategy='mean')
imputer_moda = SimpleImputer(strategy='most_frequent')

scaler_media = StandardScaler()
scaler_moda = StandardScaler()

imputer = ColumnTransformer([
    ('imputer_media', imputer_media, slice(0, 8)),
    ('imputer_moda', imputer_moda, slice(8, len(X.columns))),
 ])

scaler = ColumnTransformer([
    ('scaler_media', scaler_media, slice(0, 8)),
    ('scaler_moda', scaler_moda, slice(8, len(X.columns)))
])

pipeline = Pipeline([
    ('imputer', imputer),
    ('scaler', scaler),
    ('svm', SVC(random_state=random_state))
])

rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=random_state)  # inner
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='accuracy', cv=rskf)

scores = cross_validate(estimator=grid_search, X=X, y=y, cv=5, error_score='raise', return_estimator=True, scoring='accuracy')  # outer
print(scores['test_score'])
print(np.mean(scores['test_score']))

# HACER TAMBIEN CLASIFICADOR TONTO (DummyClassifier) para comparar los porcentajes
