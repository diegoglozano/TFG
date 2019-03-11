# Importamos paquetes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, cross_validate, cross_val_predict
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, confusion_matrix, make_scorer, roc_curve, roc_auc_score


# Variables globales
PATH = r'../../Data/pacientes_ucic_v3.csv'
RANDOM_STATE = 11
KERNEL = ['linear']
C = np.arange(-4.0, 2.0)
C = 10**C
PARAM_GRID = [{'svm__C': C, 'svm__kernel': KERNEL}]  # Hay que acceder a los parametros del pipeline como estimator__parameter
CLASS_WEIGHT = 'balanced'
SCORING = 'f1'

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
y_col = 'Situación al alta de UCI.Limitado esfuerzo terapéutico'

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
    ('svm', SVC(random_state=RANDOM_STATE, class_weight=CLASS_WEIGHT, probability=True))
])

# InnerCV (GridSearchCV de 2-folds 5-times (stratified) para obtener mejores parámetros)
rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=RANDOM_STATE)  # inner
grid_search = GridSearchCV(estimator=pipeline, param_grid=PARAM_GRID, scoring=SCORING, cv=rskf)

# OuterCV (Validación cruzada de 5 folds (stratified) para estimar Accuracy)
scores = cross_validate(estimator=grid_search, X=X, y=y, cv=5, error_score='raise', return_estimator=True, scoring=SCORING)  # outer
print('Scores: {}' .format(scores['test_score']))
print('Mean score: {}' .format(np.mean(scores['test_score'])))

# Creamos clasificador 'tonto' y obtenemos resultados también con validación cruzada (CV=5) para tener resultados más realistas
dummy_clf = DummyClassifier(strategy='most_frequent', random_state=RANDOM_STATE)
dummy_scores = cross_validate(estimator=dummy_clf, X=X, y=y, cv=5, error_score='raise', return_estimator=True, scoring=SCORING)
print('Dummy scores: {}' .format(dummy_scores['test_score']))
print('Dummy mean score: {}' .format(np.mean(dummy_scores['test_score'])))

# Predict en validación cruzada
results = cross_val_predict(grid_search, X=X, y=y, cv=5, method='predict_proba')

# ROC
fpr, tpr, thresholds = roc_curve(y, results[:, 1])
plt.figure()
plt.plot(fpr, tpr)
plt.show()

print('AUC: %.3f' % roc_auc_score(y, results[:, 1]))

distances = pd.DataFrame(list(zip(fpr, tpr, thresholds)), columns=['FPR', 'TPR', 'THRESHOLDS'])
distances['Distance'] = np.sqrt(np.power((1-distances['TPR']), 2) + np.power((distances['FPR']), 2))  # Distancia
distances.sort_values(by='Distance', ascending=True, inplace=True)
print(distances.iloc[0, :])
