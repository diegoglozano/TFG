import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score


from time import time
start = time()


# PATHS
DATA_PATH = r'../Data/pacientes_ucic_xgb.csv'

# VARIABLES
SCORING = 'accuracy'
CV_OUT = 5
CV_IN = 2
N_REPEATS = 5
RANDOM_STATE = 11
N_JOBS = -1

# DATA
df = pd.read_csv(DATA_PATH)

# X / y columns
y_col = 'Situación al alta de UCI.Precisa vigilancia herida quirúrgica'
y_cols = [col for col in df.columns if 'Situación al alta' in col]
X_cols = df.drop(y_cols, axis=1).columns

# X / y
X = df[X_cols]
y = df[y_col]

# FOLDS
skf = StratifiedKFold(n_splits=CV_OUT,
                      random_state=RANDOM_STATE)

rskf = RepeatedStratifiedKFold(n_splits=CV_IN,
                               n_repeats=N_REPEATS,
                               random_state=RANDOM_STATE)

# PARAMS
PARAMS = {
    'max_depth': [2, 3],
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [200, 300, 400],
    'scale_pos_weight': [y.value_counts()[1]/y.value_counts()[0], 1]
}

# BASE ESTIMATOR
clf = xgb.XGBClassifier(objective='binary:logistic', booster='gbtree', random_state=RANDOM_STATE, n_jobs=N_JOBS)
# INNER CV: GRID SEARCH
grid_search = GridSearchCV(clf, cv=rskf, param_grid=PARAMS, scoring=SCORING, n_jobs=N_JOBS)
# OUTER CV: cross_val_predict
preds = cross_val_predict(grid_search, X, y, cv=skf, method='predict_proba', n_jobs=N_JOBS)


def my_score(scoring, y_true, y_pred):
    if scoring == 'accuracy':
        return accuracy_score(y_true, y_pred)
    elif scoring == 'f1':
        return f1_score(y_true, y_pred)
    else:
        raise Exception(f'{scoring} no compatible. "accuracy" o "f1"')


pd.DataFrame(preds).to_csv('preds.csv', index=False)
preds = pd.read_csv('preds.csv').values
THRES = 0.5
preds = preds[:, 1] >= THRES
score = my_score(SCORING, y, preds)
print(f'Estimated {SCORING}: {score}\n')

print('Confusion matrix:')
print(confusion_matrix(y, preds, labels=[1, 0]))


end = time()
print('\n')
print(f'Total time: {end-start}')
