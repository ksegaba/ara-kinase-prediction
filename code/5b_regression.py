#!/usr/bin/env python3
"""
Regression with XGBoost or AutoGluon

Required Inputs
	-X      Path to feature matrix file
	-y_name Column name of label in X matrix
	-test   File containing list of test instances
	-save   FULL path to save output files to
	-prefix Prefix of output file names
	-alg    Algorithm to use (xgboost or autogluon)
	
	# Optional data processing arguments
	-Y      Path to label matrix file, if label not in X matrix
	-feat   File containing features (from X) to include in model
	-feat_list Comma-separated list of features (from X) to include in model
	-drop   Comma-separated list of features (from X) to drop

	# Optional feature selection arguments
	-fs     Whether to perform feature selection or not (y/n, default is n)
	-start  Starting number of features to select
	-stop   Ending number of features to select
	-step   Step size for selecting features
	-write  Write the selected features to a file (y/n, default is n)
	-type   Feature selection importance measure type (permutation/gini, default is permutation)
	
	# Optional arguments for XGBoost
	-ht     Hyperparameter tuning method (kfold/loo, default is kfold)
	-fold   k folds for Cross-Validation during training (default is 5)
	-n      Number of training repetitions (default is 10)
	-tag    Feature types/identifier for distinguising runs in results file
	-plot   Plot feature importances and predictions (default is t)

XGBoost outputs for each training repetition (prefixed with <prefix>_)
	_model_rep_*.pkl         XGBoost model
	_top20_rep_*.pdf          Plot of top 20 features' importance scores

XGboost summary outputs (prefixed with <prefix>_)
	_imp.csv                  Feature importance scores
	_preds.csv                Predicted labels
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
import time, random, pickle
import datatable as dt
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer
from autogluon.tabular import TabularPredictor
sys.path.append("./code")
from fived_feature_selection import feature_selection_clf

def parse_args():
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
		"-save", help="path to save output files to", required=True)
	req_group.add_argument(
		"-prefix", help="prefix of output file names", required=True)
	req_group.add_argument(
		"-alg", help="algorithm to use (xgboost or autogluon)", required=True)

	# Optional data processing arguments
	dp_group = parser.add_argument_group(title="Optional Data Processing")
	dp_group.add_argument(
		"-Y", help="path to label table file", default="")
	dp_group.add_argument(
		"-feat", help="file containing features (from X) to include in model",
		default="all")
	dp_group.add_argument(
		"-feat_list",
		help="comma-separated list of features (from X) to include in model",
		nargs="+", type=lambda s: [col.strip() for col in s.split(",")],
		default=[])
	dp_group.add_argument(
		"-drop", help="comma-separated list of features (from X) to drop",
		nargs="+", type=lambda s: [col.strip() for col in s.split(",")],
		default=[])
	
	# Optional feature selection arguments
	fs_group = parser.add_argument_group(title="Optional Feature Selection")
	fs_group.add_argument(
		"-fs", help="whether to perform feature selection or not (y/n)",
		default="n")
	fs_group.add_argument(
		"-start", help="starting number of features to select", type=int)
	fs_group.add_argument(
		"-stop", help="ending number of features to select", type=int)
	fs_group.add_argument(
		"-step", help="step size for selecting features", type=int)
	fs_group.add_argument(
		"-write", help="write the selected features to a file (y/n)", default="n")
	fs_group.add_argument(
		"-type", help="feature selection importance measure type (permutation/gini)",
		default="permutation")
	
	# Optional arguments for XGBoost
	xgb_group = parser.add_argument_group(title="Optional XGBoost Arguments")
	xgb_group.add_argument(
		"-ht",
		help="define the hyperparameter tuning cross-validation method (Stratified K-Fold (kfold) or Leave-One-Out(loo))",
		default="kfold")
	xgb_group.add_argument(
		"-fold",
		help="k number of cross-validation folds for training the best model. This parameter does not change the hyperopt object function fold number.",
		default=5, type=int)
	xgb_group.add_argument(
		"-n", help="number of training repetitions", default=10)
	xgb_group.add_argument(
		"-tag",
		help="Feature type/description for distinguising runs in results file",
		default="")
	xgb_group.add_argument(
		"-plot", help="plot feature importances and predictions (t/f)",
		default="t")
	
	# Help
	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit()
	args = parser.parse_args() # Read arguments from the command line
	
	return args


def hyperopt_objective_loo(params, X_train, y_train):
	"""Create the hyperparameter grid and run Hyperopt hyperparameter tuning
	with Leave-One-Out cross-validation
	"""
	
	# Create model with the current hyperparameters
	mod = xgb.XGBRegressor(
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
	
	# Normalize the data
	X_train_norm = pd.DataFrame(MinMaxScaler().fit_transform(X_train), columns=X_train.columns, index=X_train.index)
	
	# Leave-One-Out Cross-Validation
	loo = LeaveOneOut()
	loo_preds = []
	loo_actual = []
	
	for loo_train_idx, loo_val_idx in loo.split(X_train_norm):
		X_train_loo, X_val_loo = X_train_norm.iloc[loo_train_idx], X_train_norm.iloc[loo_val_idx]
		y_train_loo, y_val_loo = y_train.iloc[loo_train_idx], y_train.iloc[loo_val_idx]
		
		mod.fit(X_train_loo, y_train_loo) # Train on LOO data
		
		# Test on the LOO left-out sample
		loo_preds.append(mod.predict(X_val_loo)[0])
		loo_actual.append(y_val_loo.values[0])
		
	# Hyperopt minimizes the objective, so to maximize R^2, return
	return -r2_score(loo_actual, loo_preds)


def hyperopt_objective_kfold(params, X_train, y_train):
	"""
	Create the hyperparameter grid and run Hyperopt hyperparameter tuning
	with K-fold cross-validation
	"""
	mod = xgb.XGBRegressor(
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
	
	# Normalize the data
	X_train_norm = pd.DataFrame(MinMaxScaler().fit_transform(X_train),
		columns=X_train.columns, index=X_train.index)
	
	# Bin y_train for stratified K-fold sampling
	discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
	y_train_binned = discretizer.fit_transform(y_train.values.reshape(-1, 1)).astype(int).flatten()
	y_train_binned = pd.Series(y_train_binned, index=y_train.index)
	
	cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
	validation_loss = []
	for i, (train_idx, val_idx) in enumerate(cv.split(X_train_norm, y_train_binned)):
		X_train_cv, X_val_cv = X_train_norm.iloc[train_idx], X_train_norm.iloc[val_idx]
		y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
		
		mod.fit(X_train_cv, y_train_cv)
		cv_pred = mod.predict(X_val_cv)
		validation_loss.append(r2_score(y_val_cv, cv_pred))
	
	return -np.mean(validation_loss)


def param_hyperopt(param_grid, X_train, y_train, max_evals=100, objective_type="loo"):
	"""
	Obtain the best parameters from Hyperopt
	"""
	trials = Trials()
	
	if objective_type == "loo":
		params_best = fmin(
			fn=lambda params: hyperopt_objective_loo(params, X_train, y_train),
			space=param_grid,
			algo=tpe.suggest,
			max_evals=max_evals,
			trials=trials,
			verbose=1
		)
	
	if objective_type == "kfold":
		params_best = fmin(
			fn=lambda params: hyperopt_objective_kfold(params, X_train, y_train),
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
	
	########################## Hyperparameter tuning ###########################
	parameters = {"learning_rate":hp.uniform("learning_rate", 0.01, 0.3), # learning rate
				"max_depth":scope.int(hp.quniform("max_depth", 2, 10, 1)), # tree depth
				"subsample": hp.uniform("subsample", 0.7, 1.0), # instances per tree
				"colsample_bytree": hp.uniform("colsample_bytree", 0.8, 1.0), # features per tree
				"gamma": hp.uniform("gamma", 0.1, 5.0), # min_split_loss
				"alpha": hp.uniform("alpha", 0.1, 5.0), # L1 regularization
				"min_child_weight": scope.int(hp.quniform("min_child_weight", 5, 20, 1)), # minimum sum of instance weight needed in a child
				"n_estimators": scope.int(hp.quniform("n_estimators", 5, 400, 5)),
				"objective": "reg:squarederror",
				"eval_metric": "rmse"}
	
	start = time.time()
	best_params, trials = param_hyperopt(parameters, X_train, y_train, 100, ht)
	run_time = time.time() - start
	print("Total hyperparameter tuning time:", run_time)
	
	############# Training with Stratified K-Fold Cross-Validation #############
	results_cv = [] # hold performance metrics of cv reps
	results_test = [] # hold performance metrics on test set
	feature_imp = pd.DataFrame(index=X_train.columns)
	y_preds =  pd.DataFrame(pd.concat([y_train, y_test], axis=0).copy(deep=True))
	
	# Stratified K-Fold Cross-validation
	for j in range(0, n): # repeat cv 10 times
		print(f"Running {j+1} of {n}")
		# Build model using the best parameters
		best_model = xgb.XGBRegressor(
			learning_rate=best_params["learning_rate"],
			max_depth=int(best_params["max_depth"]),
			subsample=best_params["subsample"],
			colsample_bytree=best_params["colsample_bytree"],
			gamma=best_params["gamma"],
			alpha=best_params["alpha"],
			min_child_weight=int(best_params["min_child_weight"]),
			n_estimators=int(best_params["n_estimators"]),
			objective="reg:squarederror",
			eval_metric="rmse",
			random_state=j)
		
		X_train_norm = pd.DataFrame(MinMaxScaler().fit_transform(X_train),
			columns=X_train.columns, index=X_train.index) # Normalize
		
		# Bin y_train for stratified K-fold sampling
		discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
		y_train_binned = discretizer.fit_transform(y_train.values.reshape(-1, 1)).astype(int).flatten()
		y_train_binned = pd.Series(y_train_binned, index=y_train.index)
		
		cv = StratifiedKFold(n_splits=fold, shuffle=True, random_state=42)
		y_preds[f"cv_preds_{j}"] = np.nan
		cv_pred = []
		for i, (train_idx, val_idx) in enumerate(cv.split(X_train_norm, y_train_binned)):
			X_train_norm_cv, X_val_norm_cv = X_train_norm.iloc[train_idx], X_train_norm.iloc[val_idx]
			y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
			
			best_model.fit(X_train_norm_cv, y_train_cv)
			preds = best_model.predict(X_val_norm_cv)
			y_preds.loc[y_train.index[val_idx], f"cv_preds_{j}"] = preds
			cv_pred.append(r2_score(y_val_cv, preds))
		
		# Performance statistics on validation set
		mse_val = mean_squared_error(y_train, y_preds.loc[y_train.index, f"cv_preds_{j}"])
		rmse_val = np.sqrt(mean_squared_error(y_train, y_preds.loc[y_train.index, f"cv_preds_{j}"]))
		evs_val = explained_variance_score(y_train, y_preds.loc[y_train.index, f"cv_preds_{j}"])
		r2_val = r2_score(y_train, y_preds.loc[y_train.index, f"cv_preds_{j}"])
		cor_val = np.corrcoef(np.array(y_train), y_preds.loc[y_train.index, f"cv_preds_{j}"])
		print("Val MSE: %f" % (mse_val))
		print("Val RMSE: %f" % (rmse_val))
		print("Val EVS: %f" % (evs_val))
		print("Val R-sq: %f" % (r2_val))
		print("Val PCC: %f" % (cor_val[0, 1]))
		result_val = [mse_val, rmse_val, evs_val, r2_val, cor_val[0, 1]]
		results_cv.append(result_val)
		
		# Evaluate the model on the test set
		X_test_norm = pd.DataFrame(MinMaxScaler().fit_transform(X_test),
			columns=X_test.columns, index=X_test.index) # Normalize
		best_model.fit(X_train_norm, y_train)
		y_pred = best_model.predict(X_test_norm)
		y_preds.loc[y_test.index, f"test_preds_{j}"] = y_pred
		
		# Performance on the test set
		mse = mean_squared_error(y_test, y_pred)
		rmse = np.sqrt(mean_squared_error(y_test, y_pred))
		evs = explained_variance_score(y_test, y_pred)
		r2 = r2_score(y_test, y_pred)
		cor = np.corrcoef(np.array(y_test), y_pred)
		print(cor, y_test, y_pred)
		print("Test MSE: %f" % (mse))
		print("Test RMSE: %f" % (rmse))
		print("Test R-sq: %f" % (r2))
		print("Test PCC: %f" % (cor[0, 1]))
		result_test = [mse, rmse, evs, r2, cor[0, 1]]
		results_test.append(result_test)
		
		# Save the fitted model to a file
		filename = f"{args.save}/{prefix}_model_rep_{j}.pkl"
		pickle.dump(best_model, open(filename, "wb"))
		
		# Save feature importance scores to file
		feature_imp = pd.concat([feature_imp, pd.Series(best_model.feature_importances_,
			index=best_model.feature_names_in_, name=f"rep_{j}")],
			ignore_index=False, axis=1) 
		
		# # Save predicted labels to file
		# preds[f"rep_{j}"] = pd.concat([pd.Series(cv_pred, index=X_train.index),
		# 	pd.Series(y_pred, index=X_test.index)], axis=0)
		
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
	y_preds.to_csv(f"{args.save}/{prefix}_preds.csv")
	
	return (results_cv, results_test)


def save_xgb_results(results_cv, results_test, args, tag, run_time, nsamp, nfeats):
	"""Write training and evaluation performance results to a file."""
	
	results_cv = pd.DataFrame(
		results_cv, 
		columns=["MSE_val", "RMSE_val", "EVS_val", "R2_val", "PCC_val"])
	results_test = pd.DataFrame(
		results_test, 
		columns=["MSE_test", "RMSE_test", "EVS_test", "R2_test", "PCC_test"])
	
	# Aggregate results and save to file
	if not os.path.isfile(f"{args.save}/RESULTS_xgboost.tsv"):
		out = open(f"{args.save}/RESULTS_xgboost.tsv", "a")
		out.write("Date\tRunTime\tTag\tY\tNumInstances\tNumFeatures")
		out.write("\tCV_fold\tCV_rep\tMSE_val\tMSE_val_sd\tRMSE_val")
		out.write("\tRMSE_val_sd\tEVS_val\tEVS_val_sd")
		out.write("\tR2_val\tR2_val_sd\tPCC_val\tPCC_val_sd")
		out.write("\tMSE_test\tMSE_test_sd\tRMSE_test\tRMSE_test_sd")
		out.write("\tEVS_test\tEVS_test_sd\tR2_test\tR2_test_sd")
		out.write("\tPCC_test\tPCC_test_sd")
		out.close()
	
	out = open(f"{args.save}/RESULTS_xgboost.tsv", "a")
	out.write(f'\n{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\t')
	out.write(f"{run_time}\t{tag}\t{args.y_name}\t{nsamp}\t")
	out.write(f"{nfeats}\t{int(args.fold)}\t{int(args.n)}\t")
	out.write(f"{np.mean(results_cv.MSE_val)}\t{np.std(results_cv.MSE_val)}\t")
	out.write(f"{np.mean(results_cv.RMSE_val)}\t{np.std(results_cv.RMSE_val)}\t")
	out.write(f"{np.mean(results_cv.EVS_val)}\t{np.std(results_cv.EVS_val)}\t")
	out.write(f"{np.mean(results_cv.R2_val)}\t{np.std(results_cv.R2_val)}\t")
	out.write(f"{np.mean(results_cv.PCC_val)}\t{np.std(results_cv.PCC_val)}\t")
	out.write(f"{np.mean(results_test.MSE_test)}\t{np.std(results_test.MSE_test)}\t")
	out.write(f"{np.mean(results_test.RMSE_test)}\t{np.std(results_test.RMSE_test)}\t")
	out.write(f"{np.mean(results_test.EVS_test)}\t{np.std(results_test.EVS_test)}\t")
	out.write(f"{np.mean(results_test.R2_test)}\t{np.std(results_test.R2_test)}\t")
	out.write(f"{np.mean(results_test.PCC_test)}\t{np.std(results_test.PCC_test)}")
	out.close()


def run_autogluon(X_train, X_test, y_train, y_test, label, path, prefix):
	"""Run AutoGluon for binary classification."""
	
	# Directory to save models and other output to
	if not os.path.exists(f"{path}/{prefix}"):
		os.makedirs(f"{path}/{prefix}")
	
	os.chdir(f"{path}/{prefix}")
	
	# Normalize training and testing datasets
	X_train_norm = pd.DataFrame(MinMaxScaler().fit_transform(X_train),
		columns=X_train.columns, index=X_train.index)
	X_test_norm = pd.DataFrame(MinMaxScaler().fit_transform(X_test),
		columns=X_test.columns, index=X_test.index)
	
	# Combine X and y datasets
	X_train_norm.insert(0, label, y_train[X_train_norm.index])
	X_test_norm.insert(0, label, y_test[X_test_norm.index])
	
	# Model training
	predictor = TabularPredictor(
		label=label, eval_metric="f1", path=f"{path}/{prefix}").fit(X_train_norm)
	importance = predictor.feature_importance(X_test_norm) # permutation importance
	importance.to_csv(f"{prefix}_IMPORTANCE.csv")
	
	# Evaluation
	y_pred = predictor.predict(X_test_norm.drop(columns=[label]))
	predictor.evaluate(X_test_norm, silent=True)
	predictor.leaderboard(X_test_norm).to_csv(f"{prefix}_RESULTS.csv")


if __name__ == "__main__":
	args = parse_args()
	
	# Read in data
	X = dt.fread(args.X).to_pandas() # feature table
	X.set_index(X.columns[0], inplace=True)
	
	if args.Y == "": # get the label from X or Y files
		y = X.loc[:, args.y_name]
		X.drop(columns=args.y_name, inplace=True)
	else:
		Y = dt.fread(args.Y).to_pandas()
		Y.set_index(Y.columns[0], inplace=True)
		y = Y.loc[:, args.y_name]
	
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

	if len(args.drop) > 0:
		print("Dropping features", args.drop[0])
		X.drop(columns=args.drop[0], inplace=True)
		print(f"New dimensions: {X.shape}")
	
	# Train-test split
	X_train = X.loc[~X.index.isin(test[0])]
	y_train = y.loc[~y.index.isin(test[0])]
	try:
		X_test = X.loc[test[0]]
		y_test = y.loc[test[0]]
	except KeyError:
		print('Test set contains instances not in the feature matrix.')
		X_test = X.loc[X.index.isin(test[0])]
		y_test = y.loc[y.index.isin(test[0])]
	
	# Ensure rows are in the same order
	X_train = X_train.loc[y_train.index,:]
	X_test = X_test.loc[y_test.index,:]
	
	print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
	
	if args.fs == "y": # Run feature selection
		print("Running feature selection...")
		selected_features = feature_selection_clf(X_train, y_train,
			args.start, args.stop, args.step, args.save, args.prefix,
			args.write, args.type)
		
		for features in selected_features:
			print(f"Training model with the top {len(features)} features...")
			X_train_fs = X_train.loc[:, features]
			X_test_fs = X_test.loc[:, features]
			
			if args.alg == "xgboost":
				start = time.time()
				results_cv, results_test = run_xgb(X_train_fs, y_train,
					X_test, y_test, args.y_name, int(args.fold), int(args.n),
					f"{args.prefix}_top_{len(features)}", args.ht, args.plot)
				run_time = time.time() - start
				save_xgb_results(results_cv, results_test, args,
					f"{args.tag}_top_{len(features)}", run_time,
					len(X_train_fs), len(features))
			
			if args.alg == "autogluon":
				run_autogluon(X_train_fs, X_test_fs, y_train, y_test,
					args.y_name, args.save, f"{args.prefix}_top_{len(features)}")
	
	else: # No feature selection
		if args.alg == "xgboost":
			start = time.time()
			results_cv, results_test = run_xgb(X_train, y_train, X_test, y_test,
				args.y_name, int(args.fold), int(args.n), args.prefix, args.ht,
				args.plot)
			run_time = time.time() - start
			save_xgb_results(results_cv, results_test, args, args.tag,
				run_time, len(X_train), len(X_train.columns))
		
		if args.alg == "autogluon":
			run_autogluon(X_train, X_test, y_train, y_test, args.y_name,
				args.save, args.prefix)