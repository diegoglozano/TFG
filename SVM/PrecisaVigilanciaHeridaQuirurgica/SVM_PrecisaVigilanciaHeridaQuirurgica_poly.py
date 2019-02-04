# Importamos paquetes
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, cross_validate
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier


# Variables globales
PATH = r'../../Data/pacientes_ucic_v3.csv'
random_state = 11
kernel = ['poly']
C = np.arange(-4.0, 2.0)
C = 10**C
gamma = np.arange(-4.0, 0.0)
gamma = 10**gamma
degree = [2, 3]
coef0 = [-1]
param_grid = [{'svm__C': C, 'svm__kernel': kernel, 'svm__gamma': gamma, 'svm__degree': degree, 'svm__coef0': coef0}]  # Hay que acceder a los parametros del pipeline como estimator__parameter
class_weight = 'balanced'

# Leemos .csv
df = pd.read_csv(PATH, sep=';', index_col='Unnamed: 0')

# Guardamos los nombres de las columnas
cols = df.columns

# Guardamos los nombres de las columnas de salida
y_cols = []
for col in cols:
    y_cols.append(col) if 'Situación al alta' in col else None

# Columnas de entrada y de salida
X_cols = df.drop(y_cols, axis=1).columns
y_col = 'Situación al alta de UCI.Precisa vigilancia herida quirúrgica'

# Separamos en entrada (X) y salida (y)
X = df[X_cols]
y = df[y_col]

# Declaramos los Imputer que insertarán valores en los NaN
imputer_media = SimpleImputer(strategy='mean')
imputer_moda = SimpleImputer(strategy='most_frequent')

# Declaramos los Scaler que estandarizarán los datos
scaler_media = StandardScaler()
scaler_moda = StandardScaler()

# Creamos un ColumnTransformer para el SimpleImputer
imputer = ColumnTransformer([
    ('imputer_media', imputer_media, slice(0, 8)),
    ('imputer_moda', imputer_moda, slice(8, len(X.columns))),
 ])

# Creamos un ColumnTransformer para el StandardScaler
scaler = ColumnTransformer([
    ('scaler_media', scaler_media, slice(0, 8)),
    ('scaler_moda', scaler_moda, slice(8, len(X.columns)))
])

# Creamos el Pipeline incorporando ColumnTransformer y Clasificador
pipeline = Pipeline([
    ('imputer', imputer),
    ('scaler', scaler),
    ('svm', SVC(random_state=random_state, class_weight=class_weight))
])

# InnerCV (GridSearchCV de 2-folds 5-times (stratified) para obtener mejores parámetros)
rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=random_state)  # inner
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='accuracy', cv=rskf)

# OuterCV (Validación cruzada de 5 folds (stratified) para estimar Accuracy)
scores = cross_validate(estimator=grid_search, X=X, y=y, cv=5, error_score='raise', return_estimator=True, scoring='accuracy')  # outer
print('Scores: {}' .format(scores['test_score']))
print('Mean score: {}' .format(np.mean(scores['test_score'])))

# Creamos clasificador 'tonto' y obtenemos resultados también con validación cruzada (CV=5) para tener resultados más realistas
dummy_clf = DummyClassifier(strategy='most_frequent', random_state=random_state)
dummy_scores = cross_validate(estimator=dummy_clf, X=X, y=y, cv=5, error_score='raise', return_estimator=True, scoring='accuracy')
print('Dummy scores: {}' .format(dummy_scores['test_score']))
print('Dummy mean score: {}' .format(np.mean(dummy_scores['test_score'])))
