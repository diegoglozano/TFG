# Importamos paquetes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFECV


# Variables globales
PATH = r'../../Data/pacientes_ucic_v3.csv'
random_state = 11
kernel = ['linear']
C = np.arange(-4.0, 2.0)
C = 10**C
param_grid = [{'svm__C': C, 'svm__kernel': kernel}]  # Hay que acceder a los parametros del pipeline como estimator__parameter
class_weight = 'balanced'

# Leemos .csv
df = pd.read_csv(PATH, sep=';', index_col='Unnamed: 0')

# Guardamos los nombres de las columnas
cols = df.columns

# Guardamos los nombres de las columnas de salida
y_cols = [col for col in cols if 'Situación al alta' in col]

# Columnas de entrada y de salida
X_cols = df.drop(y_cols, axis=1).columns
y_col = 'Situación al alta de UCI.Estable no precisa cuidados especiales'

# Separamos en entrada (X) y salida (y)
X = df[X_cols]
y = df[y_col]

# DESPUES DE SELECCIONAR LAS FEATURES NO SABREMOS CUALES SON NUMERICAS Y CATEGORICAS POR LO QUE CAMBIAREMOS SU NOMBRE MEDIANTE UN DICCIONARIO PARA SABERLO
diccionario = {}

for i, col in enumerate(X.columns):
    if i < 8:
        diccionario[col] = col + ' num'
    else:
        diccionario[col] = col + ' cat'

X = X.rename(diccionario, axis=1)
X_cols = X.columns  # Actualizo X_cols

# Columnas numericas y categoricas
num_cols = [col for col in X.columns if col.endswith('num')]
cat_cols = [col for col in X.columns if col.endswith('cat')]

# Declaramos los Imputer que insertarán valores en los NaN
imputer_media = SimpleImputer(strategy='mean')
imputer_moda = SimpleImputer(strategy='most_frequent')

# Declaramos los Scaler que estandarizarán los datos
scaler_media = StandardScaler()
scaler_moda = StandardScaler()


def return_df(array):
    return pd.DataFrame(array, columns=X.columns)


# Declaramos el FunctionTransformer que incorporaremos en el Pipeline para retornar DataFrames en lugar de Numpy array
trans = FunctionTransformer(return_df, validate=False)

# Creamos un ColumnTransformer para el SimpleImputer
imputer = ColumnTransformer([
    ('imputer_media', imputer_media, num_cols),
    ('imputer_moda', imputer_moda, cat_cols)
])

# Creamos un ColumnTransformer para el StandardScaler
scaler = ColumnTransformer([
    ('scaler_media', scaler_media, num_cols),
    ('scaler_moda', scaler_moda, cat_cols)
])

# Creamos el Pipeline incorporando ColumnTransformer
pipeline = Pipeline([
    ('imputer', imputer),
    ('trans', trans),
    ('scaler', scaler),
    ('trans2', trans)
])

# TRAMPA. Problemas con el pipeline. RFE y RFECV tienen un 'check_X_y()' antes de llamar al pipeline (que contiene el imputer)
X = pipeline.fit_transform(X)

# 5 folds estratificadas
skf = StratifiedKFold(n_splits=5)

# Clasificador
clf = SVC(C=0.01, kernel='linear', random_state=random_state, class_weight=class_weight)

# RFE con cross validation
rfecv = RFECV(clf, cv=skf, scoring='accuracy')
rfecv.fit(X, y)

# Imprimimos el número de características resultante
print(f'Number of features: {rfecv.n_features_}\n')

# Imprimimos RFE accuracy
print(f'RFE accuracy: {rfecv.grid_scores_[rfecv.n_features_]}\n')

# # Dibujamos gráfica (features vs accuracy)
# # Plot number of features VS. cross-validation scores
# plt.figure()
# plt.xlabel("Number of features selected")
# plt.ylabel("Cross validation score (nb of correct classifications)")
# plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
# plt.show()


# TEST CON VALIDACIÓN CRUZADA
# Nos quedamos con las columnas correspondientes
sel_cols = X_cols[rfecv.support_]
X_sel = X[sel_cols]

# Separamos en variables numéricas y categóricas
new_num_cols = [col for col in sel_cols if col.endswith('num')]
new_cat_cols = [col for col in sel_cols if col.endswith('cat')]


def return_new_df(array):
    return pd.DataFrame(array, columns=sel_cols)


new_trans = FunctionTransformer(return_new_df, validate=False)


# Nuevo Imputer (con nuevas columnas)
new_imputer = ColumnTransformer([
    ('imputer_media', imputer_media, new_num_cols),
    ('imputer_moda', imputer_moda, new_cat_cols)
])

# Nuevo StandardScaler (con nuevas columnas)
new_scaler = ColumnTransformer([
    ('scaler_media', scaler_media, new_num_cols),
    ('scaler_moda', scaler_moda, new_cat_cols)
])

# Creamos un nuevo pipeline para hacer CV con las selected features (NUEVO)
new_pipeline = Pipeline([
    ('imputer', new_imputer),
    ('trans', new_trans),
    ('scaler', new_scaler),
    ('trans2', new_trans),
    ('svm', clf)
])

# Creamos un pipeline para hacer CV con todas las features (VIEJO)
old_pipeline = Pipeline([
    ('imputer', imputer),
    ('trans', trans),
    ('scaler', scaler),
    ('trans2', trans),
    ('svm', clf)
])

# Validación cruzada
scores_cv_sel = cross_validate(new_pipeline, X_sel, y, cv=5)  # Despues de RFE
scores_cv = cross_validate(old_pipeline, X, y, cv=5)  # Antes de RFE

# CV accuracy
print('Old CV accuracy: {}' .format(np.mean(scores_cv['test_score'])))
print('Selected features CV accuracy: {}\n' .format(np.mean(scores_cv_sel['test_score'])))


# TEST CON TRAIN/TEST
# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X_sel, y, stratify=y)

new_pipeline.fit(X_train, y_train)
scores_test = new_pipeline.predict(X_test)

# Test accuracy
print(f'Test accuracy: {np.mean(scores_test == y_test)}')
