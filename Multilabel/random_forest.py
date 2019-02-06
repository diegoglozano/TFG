import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, recall_score, precision_score


PATH = r'./Data/pacientes_ucic_v4.csv'
df = pd.read_csv(PATH)
df.drop('Situación al alta de UCI.Cuidados especiales por vía aérea artificial', axis=1, inplace=True)

y_cols = [y_col for y_col in df.columns if 'Situación al alta' in y_col]

X = df.drop(y_cols, axis=1)
y = df[y_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

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

clf = RandomForestClassifier(n_estimators=1000, max_depth=10, min_samples_split=5, min_samples_leaf=2)

# Creamos el Pipeline incorporando ColumnTransformer y Clasificador
pipeline = Pipeline([
    ('imputer', imputer),
    ('scaler', scaler),
    ('rf', clf)
])

results = cross_val_predict(estimator=pipeline, X=X, y=y, cv=5)

# Metricas
recall = recall_score(y, results, average=None)
precision = precision_score(y, results, average=None)
f1 = f1_score(y, results, average=None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
metrics = pd.DataFrame(list(zip(recall, precision, f1)), columns=['Recall', 'Precision', 'F1'], index=y.columns)

print(accuracy_score(y, results))
print(metrics)
