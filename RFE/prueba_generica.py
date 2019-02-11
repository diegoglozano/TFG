import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_transformer


sel_cols = ['1', '2', '3', '4']
print(str(sel_cols))


# Inicialmente tengo esto (2 numericas y 3 categoricas)
df = pd.DataFrame(np.random.randint(5, size=(100, 5)), columns=list('ABCDE'))

diccionario = {}

for i, col in enumerate(df.columns):
    if i < 2:
        diccionario[col] = col + ' num'
    else:
        diccionario[col] = col + ' cat'

df.rename(diccionario, axis=1, inplace=True)

num_cols = [col for col in df.columns if col.endswith('num')]
cat_cols = [col for col in df.columns if col.endswith('cat')]


def return_df(X):
    return pd.DataFrame(X, columns=df.columns)


# Creamos un ColumnTransformer para el SimpleImputer
imputer = ColumnTransformer([
    ('imputer_media', SimpleImputer(strategy='mean'), slice(0, 2)),
    ('imputer_moda', SimpleImputer(strategy='most_frequent'), slice(2, 5))
])

# Creamos un ColumnTransformer para el StandardScaler
scaler = ColumnTransformer([
    ('scaler_media', StandardScaler(), slice(0, 2)),
    ('scaler_moda', StandardScaler(), slice(2, 5))
])

# Pipeline
pipeline = Pipeline([
    ('imputer', imputer),
    ('trans', FunctionTransformer(return_df, validate=False)),
    ('scaler', scaler),
    ('trans2', FunctionTransformer(return_df, validate=False))
])

new_cols = ['B num', 'E cat']
num_new_cols = [col for col in new_cols if col.endswith('num')]
cat_new_cols = [col for col in new_cols if col.endswith('cat')]
df_new = df[new_cols]


def return_new_df(X):
    return pd.DataFrame(X, columns=df_new.columns)


new_imputer = ColumnTransformer([
    ('imputer_media', SimpleImputer(strategy='mean'), num_new_cols),
    ('imputer_moda', SimpleImputer(strategy='most_frequent'), cat_new_cols)
])

new_scaler = ColumnTransformer([
    ('scaler_media', StandardScaler(), num_new_cols),
    ('scaler_moda', StandardScaler(), cat_new_cols)
])

new_pipeline = Pipeline([
    ('new_imputer', new_imputer),
    ('trans', FunctionTransformer(return_new_df, validate=False)),
    ('new_scaler', new_scaler),
    ('trans2', FunctionTransformer(return_new_df, validate=False))
])

old = pipeline.fit_transform(df)
print(old)
new = new_pipeline.fit_transform(df)
print(new)
