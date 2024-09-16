#!/usr/bin/env python

import sys, os
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

train_path = sys.argv[1]
test_path = sys.argv[2]
label = sys.argv[3]
path = sys.argv[4]
prefix = sys.argv[5]

# Load data
X = pd.read_csv(train_path, index_col=0)
test = pd.read_csv(test_path, header=None)

# Train-test split
X_train = X.loc[~X.index.isin(test[0])]
X_test = X.loc[X.index.isin(test[0])]

# Directory to save models and other output to
if not os.path.exists(f'{path}/{prefix}'):
	os.makedirs(f'{path}/{prefix}')

os.chdir(f'{path}/{prefix}')

# Training
predictor = TabularPredictor(label=label, eval_metric='f1', path='.').fit(X_train)
importance = predictor.feature_importance(X_test) # permutation importance
importance.to_csv(f'{prefix}_IMPORTANCE.csv')

# Prediction
y_pred = predictor.predict(X_test.drop(columns=[label]))
y_pred.head()
predictor.evaluate(X_test, silent=True)
predictor.leaderboard(X_test).to_csv(f'{prefix}_RESULTS.csv')