#!/usr/bin/env python3
"""
XGBoost Binary Classification with Leave-One-Out Cross-Validation

Required Inputs
	-X      Path to feature matrix file
	-y_name Column name of label in X matrix
	-test   File containing list of test instances
	-save   Path to save output files
	-prefix Prefix of output file names
	
	# Optional
	-Y      Path to label matrix file, if label not in X matrix
	-tag    Feature types/identifier for output file naming
	-fold   k folds for Cross-Validation (default is 5)
	-n      Number of CV repetitions (default is 10)
	-feat   File containing features (from X) to include in model
	-plot   Plot feature importances and predictions (default is t)

Outputs for each training repetition (prefixed with <prefix>_)
	_lm_test_rep_*.pdf        Regression plot of predicted and actual test labels
	_model_rep_*.save         XGBoost model
	_top20_rep_*.pdf          Plot of top 20 features' importance scores

Summary outputs (prefixed with <prefix>_)
	_imp.csv                  Feature importance scores
	_cv_results.csv           Cross-validation results (various metrics)
	_test_results.csv         Evaluation results (various metrics)
	RESULTS_xgboost.txt       Aggregated results (various metrics)
 
About Hyperparameter Tuning
	Hyperparameter ranges in this script are defined for small datasets in mind.
	If you have a larger dataset, consider expanding the upper and lower bounds 
	of some hyperparameters.
	
	For example, learning_rate [0.01, 0.3], subsample [0.3, 1.0],
	colsample_bytree [0.3, 1.0], gamma [0.0, 5.0], alpha [0.0, 5.0],
	min_child_weight [1, 10, 1], n_estimators [5, 1000, 5]
"""
__author__ = "Kenia Segura Abá"

from configparser import ExtendedInterpolation
import sys, os, argparse
import time
import random
import pickle
import datatable as dt
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.pyll.base import scope
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.model_selection import LeaveOneOut, StratifiedKFold


def f1_score_safe(y_true, y_pred):
	'''Calculate the F1 score with zero division handling
	It resolves the following error:
	UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to
	no true nor predicted samples. Use `zero_division` parameter to control this
	behavior.'''
	return f1_score(y_true, y_pred, zero_division=1)

def precision_score_safe(y_true, y_pred):
	'''Calculate the precision score with zero division handling
	It resolves the following error:
	UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to
	no predicted samples. Use `zero_division` parameter to control this
	behavior.'''
	return precision_score(y_true, y_pred, zero_division=1)

def recall_score_safe(y_true, y_pred):
	'''Calculate the recall score with zero division handling
	It resolves the following error:
	UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 due to
	no true samples. Use `zero_division` parameter to control this
	behavior.'''
	return recall_score(y_true, y_pred, zero_division=1)


def hyperopt_objective_loo(params):
	"""Create the hyperparameter grid and run Hyperopt hyperparameter tuning
	with Leave-One-Out cross-validation
	"""
	
	# Create model with the current hyperparameters
	mod = xgb.XGBClassifier(
		learning_rate=params["learning_rate"],
		max_depth=int(params["max_depth"]),
		subsample=params["subsample"],
		colsample_bytree=params["colsample_bytree"],
		gamma=params["gamma"],
		alpha=params["alpha"],
		min_child_weight=params["min_child_weight"],
		n_estimators=int(params["n_estimators"]),
		objective=params["objective"],
		eval_metric=params["eval_metric"],
		random_state=42
	)
	
	# Leave-One-Out Cross-Validation
	loo = LeaveOneOut()
	loo_preds = []
	loo_actual = []
	
	for loo_train_idx, loo_val_idx in loo.split(X_train):
		X_train_loo, X_val_loo = X_train.iloc[loo_train_idx], X_train.iloc[loo_val_idx]
		y_train_loo, y_val_loo = y_train.iloc[loo_train_idx], y_train.iloc[loo_val_idx]
		
		mod.fit(X_train_loo, y_train_loo) # Train on LOO data
		
		# Test on the LOO left-out sample
		loo_preds.append(mod.predict(X_val_loo)[0])
		loo_actual.append(y_val_loo.values[0])
		
	# Average LOO Accuracy score
	loo_acc = accuracy_score(loo_actual, loo_preds)
	
	# Hyperopt will maximize the F1 score, since it minimizes the objective function loss
	return {'loss': 1-loo_acc, 'status': STATUS_OK}


def hyperopt_objective_kfold(params):
	"""
	Create the hyperparameter grid and run Hyperopt hyperparameter tuning
	with K-fold cross-validation
	Written by Thejesh Mallidi
	Modified by Kenia Segura Abá
	"""
	mod = xgb.XGBClassifier(
		learning_rate=params["learning_rate"],
		max_depth=int(params["max_depth"]),
		subsample=params["subsample"],
		colsample_bytree=params["colsample_bytree"],
		gamma=params["gamma"],
		alpha=params["alpha"],
		min_child_weight=params["min_child_weight"],
		n_estimators=int(params["n_estimators"]),
		objective=params["objective"],
		eval_metric=params["eval_metric"],
		random_state=42
	)
	
	cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
	f1_scorer = make_scorer(f1_score_safe)
	validation_loss = cross_validate(
		mod, X_train, y_train,
		scoring="accuracy",
		cv=cv,
		n_jobs=-1,
		error_score="raise"
	)
	
	# Note: Hyperopt minimizes the objective, so we want to minimize the loss, thereby maximizing the F1 score
	return -np.mean(validation_loss["test_score"])


def param_hyperopt(param_grid, max_evals=100, objective_type="loo"):
	"""
	Obtain the best parameters from Hyperopt
	Written by Thejesh Mallidi
	"""
	trials = Trials()

	if objective_type == "loo":
		params_best = fmin(
			fn=hyperopt_objective_loo,
			space=param_grid,
			algo=tpe.suggest,
			max_evals=max_evals,
			trials=trials,
			verbose=1
		)
	
	if objective_type == "kfold":
		params_best = fmin(
			fn=hyperopt_objective_kfold,
			space=param_grid,
			algo=tpe.suggest,
			max_evals=max_evals,
			trials=trials,
			verbose=1
		)
	
	print("\n\nBest parameters:", params_best)
	return params_best, trials


def run_xgb(X_train, y_train, X_test, y_test, trait, fold, n, prefix, ht, plot): # I want to try implementing xgb.train method
	""" Train XGBoost Classification Model """
	print(f"Training model for {trait}...")
	
	###### Hyperparameter tuning with leave-one-out cross-validation #######
	parameters = {"learning_rate":hp.uniform("learning_rate", 0.01, 0.2), # learning rate
				"max_depth":scope.int(hp.quniform("max_depth", 2, 6, 1)), # tree depth
				"subsample": hp.uniform("subsample", 0.7, 1.0), # instances per tree
				"colsample_bytree": hp.uniform("colsample_bytree", 0.8, 1.0), # features per tree
				"gamma": hp.uniform("gamma", 0.1, 5.0), # min_split_loss
				"alpha": hp.uniform("alpha", 0.1, 5.0), # L1 regularization
				"min_child_weight": scope.int(hp.quniform("min_child_weight", 5, 20, 1)), # minimum sum of instance weight needed in a child
				"n_estimators": scope.int(hp.quniform("n_estimators", 5, 400, 5)),
				"objective": "binary:logistic",
				"eval_metric": "logloss"}
	
	start = time.time()
	best_params, trials = param_hyperopt(parameters, 100, ht)
	run_time = time.time() - start
	print("Total hyperparameter tuning time:", run_time)
	
	############# Training with Stratified K-Fold Cross-Validation #############
	results_cv = [] # hold performance metrics of cv reps
	results_test = [] # hold performance metrics on test set
	feature_imp = pd.DataFrame(index=X_train.columns)
	preds = {}
	
	# Stratified K-Fold Cross-validation
	for j in range(0, n): # repeat cv 10 times
		print(f"Running {j+1} of {n}")
		# Build model using the best parameters
		best_model = xgb.XGBClassifier(
			eta=best_params["learning_rate"],
			max_depth=int(best_params["max_depth"]),
			subsample=best_params["subsample"],
			colsample_bytree=best_params["colsample_bytree"],
			gamma=best_params["gamma"],
			alpha=best_params["alpha"],
			min_child_weight=int(best_params["min_child_weight"]),
			n_estimators=int(best_params["n_estimators"]),
			objective="binary:logistic",
			eval_metric="logloss",
			random_state=j)
		
		# ####### check if model is learning anything #######
		# best_model.fit(X_train, y_train)
		# y_pred = best_model.predict(X_train)
		# print(y_pred)
		# roc_auc_train = roc_auc_score(y_train, y_pred)
		# prec_train = precision_score_safe(y_train, y_pred)
		# reca_train = recall_score_safe(y_train, y_pred)
		# f1_train = f1_score_safe(y_train, y_pred)
		# mcc_train = matthews_corrcoef(y_train, y_pred)
		# acc_train = accuracy_score(y_train, y_pred)
		# print("Train ROC-AUC: %f" % (roc_auc_train))
		# print("Train Precision: %f" % (prec_train))
		# print("Train Recall: %f" % (reca_train))
		# print("Train F1: %f" % (f1_train))
		# print("Train MCC: %f" % (mcc_train))
		# print("Train Accuracy: %f" % (acc_train))
		####################################################

		k_fold = StratifiedKFold(n_splits=fold, shuffle=True, random_state=j)
		cv_pred = cross_val_predict(
			best_model, X_train, y_train, cv=k_fold, n_jobs=-1)
		
		# Thejesh recommended I look at the individual validation performances #
		# if j==0:
		# 	for train_idx, val_idx in k_fold.split(X_train, y_train):
		# 		print('y_train', y_train.iloc[train_idx].value_counts())
		# 		print('X_train', y_train.iloc[val_idx].value_counts())
				
		# 		best_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
		# 		best_model.predict(X_train.iloc[val_idx])
		########################################################################

		# Performance statistics on validation set
		roc_auc_val = roc_auc_score(y_train, cv_pred) # Not defined for Leave-One-Out
		prec_val = precision_score_safe(y_train, cv_pred)  # Precision not defined and set to 0 error
		reca_val = recall_score_safe(y_train, cv_pred)
		f1_val = f1_score_safe(y_train, cv_pred)
		mcc_val = matthews_corrcoef(y_train, cv_pred)
		acc_val = accuracy_score(y_train, cv_pred)
		print("Val ROC-AUC: %f" % (roc_auc_val))
		print("Val Precision: %f" % (prec_val))
		print("Val Recall: %f" % (reca_val))
		print("Val F1: %f" % (f1_val))
		print("Val MCC: %f" % (mcc_val))
		print("Val Accuracy: %f" % (acc_val))
		result_val = [roc_auc_val, prec_val, reca_val, f1_val, mcc_val, acc_val]
		results_cv.append(result_val)
		
		# Evaluate the model on the test set
		best_model.fit(X_train, y_train)
		y_pred = best_model.predict(X_test)
		
		# Performance on the test set
		roc_auc_test = roc_auc_score(y_test, y_pred)
		prec_test = precision_score_safe(y_test, y_pred)
		reca_test = recall_score_safe(y_test, y_pred)
		f1_test = f1_score_safe(y_test, y_pred)
		mcc_test = matthews_corrcoef(y_test, y_pred)
		acc_test = accuracy_score(y_test, y_pred)
		print("Test ROC-AUC: %f" % (roc_auc_test))
		print("Test Precision: %f" % (prec_test))
		print("Test Recall: %f" % (reca_test))
		print("Test F1: %f" % (f1_test))
		print("Test MCC: %f" % (mcc_test))
		print("Test Accuracy: %f" % (acc_test))
		result_test = [roc_auc_test, prec_test, reca_test,
			f1_test, mcc_test, acc_test]
		results_test.append(result_test)
		
		# Save the fitted model to a file
		filename = f"{args.save}/{prefix}_model_rep_{j}.pkl"
		pickle.dump(best_model, open(filename, "wb"))
		
		# Save feature importance scores to file
		feature_imp = pd.concat([feature_imp, pd.Series(best_model.feature_importances_,
			index=best_model.feature_names_in_, name=f"rep_{j}")],
			ignore_index=False, axis=1) 
		
		# Save predicted labels to file
		preds[f"rep_{j}"] = pd.concat([pd.Series(cv_pred, index=X_train.index),
			pd.Series(y_pred, index=X_test.index)], axis=0)

		if plot=="t":
			# Plot feature importances
			xgb.plot_importance(
				best_model, grid=False, max_num_features=20, 
				title=f"{trait} Feature Importances", xlabel="Weight")
			plt.tight_layout()
			plt.savefig(f"{args.save}/{prefix}_top20_rep_{j}.pdf", format="pdf")
			plt.close()
	
	# Write feature importances across reps to file
	feature_imp.to_csv(f"{args.save}/{prefix}_imp.csv")
	
	# Write predictions across reps to file
	pd.DataFrame.from_dict(preds).to_csv(f"{args.save}/{prefix}_preds.csv")

	tmp = pd.DataFrame.from_dict(preds)
	preds_train = tmp.loc[X_train.index,:]
	f1_train = preds_train.apply(lambda x: f1_score_safe(y_train, x), axis=1)
	roc_auc_train = preds_train.apply(lambda x: roc_auc_score(y_train, x), axis=1)
	prec_train = preds_train.apply(lambda x: precision_score_safe(y_train, x), axis=1)
	reca_train = preds_train.apply(lambda x: recall_score_safe(y_train, x), axis=1)
	mcc_train = preds_train.apply(lambda x: matthews_corrcoef(y_train, x), axis=1)
	acc_train = preds_train.apply(lambda x: accuracy_score(y_train, x), axis=1)
	print(f"Train F1: {f1_train.mean()}")
	print(f"Train ROC-AUC: {roc_auc_train.mean()}")
	print(f"Train Precision: {prec_train.mean()}")
	print(f"Train Recall: {reca_train.mean()}")
	print(f"Train MCC: {mcc_train.mean()}")
	print(f"Train Accuracy: {acc_train.mean()}")
	
	return (results_cv, results_test)


if __name__ == "__main__":
	# Argument parser
	parser = argparse.ArgumentParser(
		description="XGBoost Regression on SNP and ORF data")
	
	# Required input
	req_group = parser.add_argument_group(title="Required Input")
	req_group.add_argument(
		"-X", help="path to feature table file", required=True)
	req_group.add_argument(
		"-y_name", help="name of label in X file", required=True)
	req_group.add_argument(
		"-test", help="path to file of test set instances", required=True)
	req_group.add_argument(
		"-save", help="path to save output files", required=True)
	req_group.add_argument(
		"-prefix", help="prefix of output file names", required=True)
	
	# Optional input
	req_group.add_argument(
		"-Y", help="path to label table file", default="")
	req_group.add_argument(
		"-tag", help="description about run to add to results file", default="")
	req_group.add_argument(
		"-ht", help="define the hyperparameter tuning cross-validation method (Stratified K-Fold (kfold) or Leave-One-Out(loo))",
		default="kfold")
	req_group.add_argument(
		"-fold",
		help="k number of cross-validation folds for training the best model. This parameter does not change the hyperopt object function fold number.",
		default=5)
	req_group.add_argument(
		"-n", help="number of cross-validation repetitions", default=10)
	req_group.add_argument(
		"-feat", help="file containing features (from X) to include in model", default="all")
	req_group.add_argument(
		"-feat_list", help="comma-separated list of features (from X) to include in model", 
		nargs="+", type=lambda s: [col.strip() for col in s.split(",")], default=[])
	req_group.add_argument(
		"-plot", help="plot feature importances and predictions (t/f)", default="t")
	
	# Help
	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit()
	args = parser.parse_args() # Read arguments

	# Read in data
	X = dt.fread(args.X).to_pandas() # feature table
	X.set_index(X.columns[0], inplace=True)

	if args.Y == "": # get the label from X or Y files
		y = X.loc[:, args.y_name]
		X.drop(columns=args.y_name, inplace=True)
	else:
		Y = args.Y
		y = Y.loc[:, args.y_name]
  
	y = y.astype(int) # convert binary bool values to integer
	
	test = pd.read_csv(args.test, header=None) # test instances

	# Filter out features not in the given feat file - default: keep all
	if args.feat != "all":
		print("Using subset of features from: %s" % args.feat)
		with open(args.feat) as f:
			features = f.read().strip().splitlines()
		X = X.loc[:,features]
		print(f"New dimensions: {X.shape}")
	
	if len(args.feat_list) > 0:
		print("Using subset of features from list", args.feat_list[0])
		X = X.loc[:,args.feat_list[0]]
		print(f"New dimensions: {X.shape}")

	# Train-test split
	X_train = X.loc[~X.index.isin(test[0])]
	X_test = X.loc[test[0]]
	y_train = y.loc[~y.index.isin(test[0])]
	y_test = y.loc[test[0]]

	# Ensure rows are in the same order
	X_train = X_train.loc[y_train.index,:]
	X_test = X_test.loc[y_test.index,:]

	print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
	print(y_train.value_counts(), y_test.value_counts())
	
	# Train the model with Leave-One-Out Testing
	start = time.time()
	results_cv, results_test = run_xgb(X_train, y_train, X_test, y_test,
		args.y_name, int(args.fold), int(args.n), args.prefix, args.ht, args.plot)
	run_time = time.time() - start
	print("Training Run Time: %f" % (run_time))

	# Save results to file
	results_cv = pd.DataFrame(
		results_cv, 
		columns=["ROC-AUC_val", "Precision_val", "Recall_val", "F1_val",
		"MCC_val", "Accuracy_val"])
	results_test = pd.DataFrame(
		results_test, 
		columns=["ROC-AUC_test", "Precision_test", "Recall_test", "F1_test",
		"MCC_test", "Accuracy_test"])

	# Aggregate results and save to file
	if not os.path.isfile(f"{args.save}/RESULTS_xgboost.txt"):
		out = open(f"{args.save}/RESULTS_xgboost.txt", "a")
		out.write("Date\tRunTime\tTag\tY\tNumInstances\tNumFeatures")
		out.write("\tCV_fold\tCV_rep\tROC-AUC_val\tROC-AUC_val_sd\tPrecision_val")
		out.write("\tPrecision_val_sd\tRecall_val\tRecall_val_sd")
		out.write("\tF1_val\tF1_val_sd\tMCC_val\tMCC_val_sd")
		out.write("\tAccuracy_val\tAccuracy_val_sd\tROC-AUC_test\tROC-AUC_test_sd")
		out.write("\tPrecision_test\tPrecision_test_sd\tRecall_test")
		out.write("\tRecall_test_sd\tF1_test\tF1_test_sd\tMCC_test")
		out.write("\tMCC_test_sd\tAccuracy_test\tAccuracy_test_sd")
		out.close()

	out = open(f"{args.save}/RESULTS_xgboost.txt", "a")
	out.write(f'\n{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\t')
	out.write(f'{run_time}\t{args.tag}\t{args.y_name}\t{X_train.shape[0]}\t')
	out.write(f'{X.shape[1]}\t{int(args.fold)}\t{int(args.n)}\t')
	out.write(f'{np.mean(results_cv["ROC-AUC_val"])}\t{np.std(results_cv["ROC-AUC_val"])}\t')
	out.write(f'{np.mean(results_cv["Precision_val"])}\t{np.std(results_cv["Precision_val"])}\t')
	out.write(f'{np.mean(results_cv["Recall_val"])}\t{np.std(results_cv["Recall_val"])}\t')
	out.write(f'{np.mean(results_cv["F1_val"])}\t{np.std(results_cv["F1_val"])}\t')
	out.write(f'{np.mean(results_cv["MCC_val"])}\t{np.std(results_cv["MCC_val"])}\t')
	out.write(f'{np.mean(results_cv["Accuracy_val"])}\t{np.std(results_cv["Accuracy_val"])}\t')
	out.write(f'{np.mean(results_test["ROC-AUC_test"])}\t{np.std(results_test["ROC-AUC_test"])}\t')
	out.write(f'{np.mean(results_test["Precision_test"])}\t{np.std(results_test["Precision_test"])}\t')
	out.write(f'{np.mean(results_test["Recall_test"])}\t{np.std(results_test["Recall_test"])}\t')
	out.write(f'{np.mean(results_test["F1_test"])}\t{np.std(results_test["F1_test"])}\t')
	out.write(f'{np.mean(results_test["MCC_test"])}\t{np.std(results_test["MCC_test"])}\t')
	out.write(f'{np.mean(results_test["Accuracy_test"])}\t{np.std(results_test["Accuracy_test"])}')
