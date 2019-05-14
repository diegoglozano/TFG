import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score, confusion_matrix

from XGB_ROC.my_functions import my_label_encoder


DATA_PATH = r'../Data/pacientes_ucic_xgb.csv'
RANDOM_STATE = 11
CV = 5

df = pd.read_csv(DATA_PATH)

y_cols = [col for col in df.columns if 'Situación al alta' in col]
y_col = 'Situación al alta de UCI.Vigilancia anemia/anticoagulantes'
X = df.drop(y_cols, axis=1)
y = df[y_col]

SCALE_POS_WEIGHT = y.value_counts()[1]/y.value_counts()[0]

params = {
    'max_depth': 2,
    'subsample': 0.1,
    'objective': 'binary:hinge',
    'eta': 0.001,
    'silent': True,
    'seed': RANDOM_STATE,
    # 'scale_pos_weight': SCALE_POS_WEIGHT
}


def my_f1_score(self_preds, self_dtrain):
    labels = self_dtrain.get_label()
    return 'f1-score', f1_score(labels, self_preds)


skf = StratifiedKFold(n_splits=CV, random_state=RANDOM_STATE)
dtrain = xgb.DMatrix(X, y)
results = xgb.cv(params, dtrain, num_boost_round=20, folds=skf, feval=my_f1_score)
print(results[['train-f1-score-mean', 'test-f1-score-mean']])

clf = xgb.XGBClassifier(
    max_depth=2,
    subsample=0.01,
    objective='binary:hinge',
    learning_rate=0.001,
    random_state=RANDOM_STATE
)

scores = cross_val_score(clf, X, y, cv=skf)

print(np.mean(scores))
