import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import hamming_loss, accuracy_score
from sklearn.impute import SimpleImputer

from skmultilearn.ensemble import RakelD, RakelO


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

# Creamos el Pipeline incorporando ColumnTransformer y Clasificador
pipeline = Pipeline([
    ('imputer', imputer),
    ('scaler', scaler)
])

pipeline.fit(X_train)
X_train = pipeline.transform(X_train)
X_test = pipeline.transform(X_test)

# PROBAMOS CLASIFICADORES
clf = RakelD(
    base_classifier=SVC(kernel='linear', class_weight='balanced'),
    base_classifier_require_dense=[True, True],
    labelset_size=1000
)

clf.fit(X_train, y_train)
train_pred = clf.predict(X_train)
test_pred = clf.predict(X_test)

ham_loss = hamming_loss(y_test, test_pred)
acc = accuracy_score(y_test, test_pred)
print(ham_loss)
print(acc)
