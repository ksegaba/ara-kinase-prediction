#!/usr/bin/env python3
"""
Binary Classification with XGBoost or AutoGluon

Required Inputs
	-X      Path to feature matrix file
	-y_name Column name of label in X matrix
	-save   FULL path to save output files to
	-prefix Prefix of output file names
	-alg    Algorithm to use (xgboost or autogluon)
	
	# Optional data processing arguments
	-Y      Path to label matrix file, if label not in X matrix
	-feat   File containing features (from X) to include in model
	-feat_list Comma-separated list of features (from X) to include in model
	-bal    Balance the training set (y/n, default is n)
	-n_bal  Number of balanced datasets to create (default is 15)
	-test   File containing list of test instances
	-size   Size of test set (default is 1/11 of the data; include if no test file is provided)
	
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
	RESULTS_xgboost.tsv       Aggregated results (various metrics)

About Hyperparameter Tuning
	Hyperparameter ranges in this script are defined for small datasets in mind.
	If you have a larger dataset, consider expanding the upper and lower bounds 
	of some hyperparameters.
	
	For example, learning_rate [0.01, 0.3], subsample [0.3, 1.0],
	colsample_bytree [0.3, 1.0], gamma [0.0, 5.0], alpha [0.0, 5.0],
	min_child_weight [1, 10, 1], n_estimators [5, 1000, 5]
"""

from fived_feature_selection import feature_selection_clf
__author__ = "Kenia Segura Abá"

from configparser import ExtendedInterpolation
import sys
import os
import argparse
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
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, train_test_split
from autogluon.tabular import TabularPredictor
sys.path.append('./code')


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
        "-cl_list", help="list of class categories if they are not binary (0 & 1)",
        nargs="+", type=lambda s: [cl.strip() for cl in s.split(",")],
        default=[0, 1])  # default is binary classification
    dp_group.add_argument(
        "-feat", help="file containing features (from X) to include in model",
        default="all")
    dp_group.add_argument(
        "-feat_list",
        help="comma-separated list of features (from X) to include in model",
        nargs="+", type=lambda s: [col.strip() for col in s.split(",")],
        default=[])
    dp_group.add_argument(
        "-bal", help="whether to balance the training set or not (y/n)",
        default="n")
    dp_group.add_argument(
        "-n_bal", help="number of balanced datasets to create", default=15, type=int)
    dp_group.add_argument(
        "-test", help="path to file of test set instances", default="", type=str)
    dp_group.add_argument(
        "-size", help="size of test set (default is 1/11 of the data)", default=1/11, type=float)

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
    args = parser.parse_args()  # Read arguments from the command line

    return args


def create_balanced(X, n, y_name):
    '''Create balanced training datasets by downsampling the majority class.'''

    # Split the data into positive and negative classes
    pos_class = X[X[y_name] == 1]
    neg_class = X[X[y_name] == 0]

    num_datasets = n
    balanced_datasets = []
    for b in range(num_datasets):
        # Downsample the negative class to match the number of positives
        neg_downsampled = resample(neg_class,
                                   replace=False,  # no replacement
                                   # match positive class size
                                   n_samples=len(pos_class),
                                   random_state=b)

        # Combine the positive and downsampled negative class to create a balanced dataset
        balanced_data = pd.concat([pos_class, neg_downsampled])

        # Shuffle the dataset to mix positive and negative samples
        balanced_data = balanced_data.sample(frac=1, random_state=b)

        # Append this balanced dataset to the list
        balanced_datasets.append(balanced_data)

    return balanced_datasets


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


def hyperopt_objective_loo(params, X_train, y_train):
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

    # Normalize the data
    X_train_norm = MinMaxScaler().fit_transform(X_train)

    # Leave-One-Out Cross-Validation
    loo = LeaveOneOut()
    loo_preds = []
    loo_actual = []

    for loo_train_idx, loo_val_idx in loo.split(X_train_norm):
        X_train_loo, X_val_loo = X_train_norm.iloc[loo_train_idx], X_train_norm.iloc[loo_val_idx]
        y_train_loo, y_val_loo = y_train.iloc[loo_train_idx], y_train.iloc[loo_val_idx]

        mod.fit(X_train_loo, y_train_loo)  # Train on LOO data

        # Test on the LOO left-out sample
        loo_preds.append(mod.predict(X_val_loo)[0])
        loo_actual.append(y_val_loo.values[0])

    # Average LOO Accuracy score
    loo_acc = accuracy_score(loo_actual, loo_preds)

    # Hyperopt will maximize the accuracy, since it minimizes the objective function loss


def hyperopt_objective_kfold(params, X_train, y_train):
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

    # Normalize the data
    X_train_norm = MinMaxScaler().fit_transform(X_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scorer = make_scorer(f1_score_safe)
    validation_f1 = cross_validate(
        mod, X_train_norm, y_train,
        scoring=f1_scorer,
        cv=cv,
        n_jobs=-1,
        error_score="raise"
    )

    # Note: Hyperopt minimizes the objective, so we want to minimize the loss, thereby maximizing the F1 score
    f1 = np.mean(validation_f1["test_score"])
    return {'loss': 1-f1, 'status': STATUS_OK}


def param_hyperopt(param_grid, X_train, y_train, max_evals=100, objective_type="loo"):
    """
    Obtain the best parameters from Hyperopt
    Written by Thejesh Mallidi
    Modified by Kenia Segura Abá
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
            fn=lambda params: hyperopt_objective_kfold(
                params, X_train, y_train),
            space=param_grid,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            verbose=1
        )

    print("\n\nBest parameters:", params_best)
    return params_best, trials


# I want to try implementing xgb.train method
def run_xgb(X_train, y_train, X_test, y_test, trait, fold, n, prefix, ht, plot):
    """ Train XGBoost Classification Model """
    print(f"Training model for {trait}...")

    ############# Training with Stratified K-Fold Cross-Validation #############
    results_cv = []  # hold performance metrics of cv reps
    results_test = []  # hold performance metrics on test set
    feature_imp = pd.DataFrame(index=X_train.columns)
    preds = {}

    for j in range(0, n):  # repeat cv 10 times
        print(f"Running {j+1} of {n}")
        ###### Hyperparameter tuning with leave-one-out cross-validation #######
        # Hyperparameter grid for a smaller dataset
        # parameters = {"learning_rate":hp.uniform("learning_rate", 0.01, 0.2), # learning rate
        # 			"max_depth":scope.int(hp.quniform("max_depth", 2, 6, 1)), # tree depth
        # 			"subsample": hp.uniform("subsample", 0.7, 1.0), # instances per tree
        # 			"colsample_bytree": hp.uniform("colsample_bytree", 0.8, 1.0), # features per tree
        # 			"gamma": hp.uniform("gamma", 0.1, 5.0), # min_split_loss
        # 			"alpha": hp.uniform("alpha", 0.1, 5.0), # L1 regularization
        # 			"min_child_weight": scope.int(hp.quniform("min_child_weight", 5, 20, 1)), # minimum sum of instance weight needed in a child
        # 			"n_estimators": scope.int(hp.quniform("n_estimators", 5, 400, 5)),
        # 			"objective": "binary:logistic",
        # 			"eval_metric": "logloss"}

        # Hyperparameter grid for a larger dataset
        parameters = {"learning_rate": hp.uniform("learning_rate", 0.01, 0.4),  # learning rate
                      # tree depth
                      "max_depth": scope.int(hp.quniform("max_depth", 2, 10, 1)),
                      # instances per tree
                      "subsample": hp.uniform("subsample", 0.2, 1.0),
                      # features per tree
                      "colsample_bytree": hp.uniform("colsample_bytree", 0.2, 1.0),
                      # min_split_loss
                      "gamma": hp.uniform("gamma", 0.0, 6.0),
                      # L1 regularization
                      "alpha": hp.uniform("alpha", 0.0, 6.0),
                      # minimum sum of instance weight needed in a child
                      "min_child_weight": scope.int(hp.quniform("min_child_weight", 1, 10, 1)),
                      "n_estimators": scope.int(hp.quniform("n_estimators", 5, 500, 5)),
                      "objective": "binary:logistic",
                      "eval_metric": "logloss"}

        start = time.time()
        best_params, trials = param_hyperopt(
            parameters, X_train, y_train, 100, ht)
        run_time = time.time() - start
        print("Total hyperparameter tuning time:", run_time)

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

        X_train_norm = pd.DataFrame(MinMaxScaler().fit_transform(X_train),
                                    columns=X_train.columns, index=X_train.index)  # Normalize features

        k_fold = StratifiedKFold(n_splits=fold, shuffle=True, random_state=j)
        cv_pred = cross_val_predict(
            # Stratified K-Fold Cross-Validation
            best_model, X_train_norm, y_train, cv=k_fold, n_jobs=-1)

        # Performance statistics on validation set
        # Not defined for Leave-One-Out
        roc_auc_val = roc_auc_score(y_train, cv_pred)
        # Precision not defined and set to 0 error
        prec_val = precision_score_safe(y_train, cv_pred)
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
        result_val = [roc_auc_val, prec_val,
                      reca_val, f1_val, mcc_val, acc_val]
        results_cv.append(result_val)

        # Evaluate the model on the test set
        X_test_norm = MinMaxScaler().fit_transform(X_test)  # Normalize
        best_model.fit(X_train_norm, y_train)
        y_pred = best_model.predict(X_test_norm)

        if len(y_test.unique()) > 1:
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
        else:
            # If the test set has only one class, we cannot calculate these metrics
            roc_auc_test = prec_test = reca_test = f1_test = mcc_test = acc_test = -1

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

        if plot == "t":
            # Plot feature importances
            xgb.plot_importance(
                best_model, grid=False, max_num_features=20,
                title=f"{trait} Feature Importances", xlabel="Weight")
            plt.tight_layout()
            plt.savefig(
                f"{args.save}/{prefix}_top20_rep_{j}.pdf", format="pdf")
            plt.close()

    # Write feature importances across reps to file
    feature_imp.to_csv(f"{args.save}/{prefix}_imp.csv")

    # Write predictions across reps to file
    pd.DataFrame.from_dict(preds).to_csv(f"{args.save}/{prefix}_preds.csv")

    return (results_cv, results_test)


def save_xgb_results(results_cv, results_test, args, tag, run_time, nsamp, nfeats):
    '''Write training and evaluation performance results to a file.'''

    results_cv = pd.DataFrame(
        results_cv,
        columns=["ROC-AUC_val", "Precision_val", "Recall_val", "F1_val",
                 "MCC_val", "Accuracy_val"])
    results_test = pd.DataFrame(
        results_test,
        columns=["ROC-AUC_test", "Precision_test", "Recall_test", "F1_test",
                 "MCC_test", "Accuracy_test"])

    # Aggregate results and save to file
    if not os.path.isfile(f"{args.save}/RESULTS_xgboost.tsv"):
        out = open(f"{args.save}/RESULTS_xgboost.tsv", "a")
        out.write("Date\tRunTime\tTag\tY\tNumInstances\tNumFeatures")
        out.write("\tCV_fold\tCV_rep\tROC-AUC_val\tROC-AUC_val_sd\tPrecision_val")
        out.write("\tPrecision_val_sd\tRecall_val\tRecall_val_sd")
        out.write("\tF1_val\tF1_val_sd\tMCC_val\tMCC_val_sd")
        out.write("\tAccuracy_val\tAccuracy_val_sd\tROC-AUC_test\tROC-AUC_test_sd")
        out.write("\tPrecision_test\tPrecision_test_sd\tRecall_test")
        out.write("\tRecall_test_sd\tF1_test\tF1_test_sd\tMCC_test")
        out.write("\tMCC_test_sd\tAccuracy_test\tAccuracy_test_sd")
        out.close()

    out = open(f"{args.save}/RESULTS_xgboost.tsv", "a")
    out.write(f'\n{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\t')
    out.write(f'{run_time}\t{tag}\t{args.y_name}\t{nsamp}\t')
    out.write(f'{nfeats}\t{int(args.fold)}\t{int(args.n)}\t')
    out.write(
        f'{np.mean(results_cv["ROC-AUC_val"])}\t{np.std(results_cv["ROC-AUC_val"])}\t')
    out.write(
        f'{np.mean(results_cv["Precision_val"])}\t{np.std(results_cv["Precision_val"])}\t')
    out.write(
        f'{np.mean(results_cv["Recall_val"])}\t{np.std(results_cv["Recall_val"])}\t')
    out.write(
        f'{np.mean(results_cv["F1_val"])}\t{np.std(results_cv["F1_val"])}\t')
    out.write(
        f'{np.mean(results_cv["MCC_val"])}\t{np.std(results_cv["MCC_val"])}\t')
    out.write(
        f'{np.mean(results_cv["Accuracy_val"])}\t{np.std(results_cv["Accuracy_val"])}\t')
    out.write(
        f'{np.mean(results_test["ROC-AUC_test"])}\t{np.std(results_test["ROC-AUC_test"])}\t')
    out.write(
        f'{np.mean(results_test["Precision_test"])}\t{np.std(results_test["Precision_test"])}\t')
    out.write(
        f'{np.mean(results_test["Recall_test"])}\t{np.std(results_test["Recall_test"])}\t')
    out.write(
        f'{np.mean(results_test["F1_test"])}\t{np.std(results_test["F1_test"])}\t')
    out.write(
        f'{np.mean(results_test["MCC_test"])}\t{np.std(results_test["MCC_test"])}\t')
    out.write(
        f'{np.mean(results_test["Accuracy_test"])}\t{np.std(results_test["Accuracy_test"])}')
    out.close()


def run_autogluon(X_train, X_test, y_train, y_test, label, path, prefix):
    '''Run AutoGluon for binary classification.'''

    # Directory to save models and other output to
    if not os.path.exists(f'{path}/{prefix}'):
        os.makedirs(f'{path}/{prefix}')

    os.chdir(f'{path}/{prefix}')

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
        label=label, eval_metric='f1', path=f'{path}/{prefix}').fit(X_train_norm)
    importance = predictor.feature_importance(
        X_test_norm)  # permutation importance
    importance.to_csv(f'{prefix}_IMPORTANCE.csv')

    # Evaluation
    y_pred = predictor.predict(X_test_norm.drop(columns=[label]))
    print(y_pred)
    predictor.evaluate(X_test_norm, silent=True)
    predictor.leaderboard(X_test_norm).to_csv(f'{prefix}_RESULTS.csv')


if __name__ == "__main__":
    args = parse_args()

    # Read in data
    X = dt.fread(args.X).to_pandas()  # feature table
    X.set_index(X.columns[0], inplace=True)

    if args.Y == "":  # get the label from X or Y files
        y = X.loc[:, args.y_name]
        X.drop(columns=args.y_name, inplace=True)
    else:
        Y = args.Y
        y = Y.loc[:, args.y_name]
        if y.isna().any():
            print("Removing rows with missing labels")
            y = y.dropna()
            X = X.loc[y.index, :]
            print(f"New dimensions X: {X.shape} and y: {y.shape}")

    # Filter out features not in the given feat file - default: keep all
    if args.feat != "all":
        print("Using subset of features from: %s" % args.feat)
        with open(args.feat) as f:
            features = f.read().strip().splitlines()
        X = X.loc[:, features]
        print(f"New dimensions: {X.shape}")

    if len(args.feat_list) > 0:
        print("Using subset of features from list", args.feat_list[0])
        X = X.loc[:, args.feat_list[0]]
        print(f"New dimensions: {X.shape}")

    # Train-test split
    if args.test != '':
        test = dt.fread(args.test, header=False).to_pandas()  # test instances
        if len(test.columns) > 1:  # Features were included in the test file
            # re-read the test file, include header
            test = dt.fread(args.test).to_pandas()
            test.set_index(test.columns[0], inplace=True)
            y_train = y.copy(deep=True)
            X_train = X.copy(deep=True)
            y_test = test.loc[:, args.y_name]
            X_test = test.drop(columns=args.y_name).astype(int)
            del X, y
        else:
            X_train = X.loc[~X.index.isin(test.iloc[:, 0])]
            X_test = X.loc[test.iloc[:, 0]]
            y_train = y.loc[~y.index.isin(test.iloc[:, 0])]
            y_test = y.loc[test.iloc[:, 0]]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.size, random_state=2305, stratify=y)

    # Convert classes to binary if not already
    y_class_map = {}
    if list(y_train.unique()) != [0, 1]:
        y_class_map[args.cl_list[0][0]] = 0
        y_class_map[args.cl_list[0][1]] = 1

    print("New classes:", y_class_map)
    y_train = y_train.replace(y_class_map)
    print(y_train, y_test)

    # Ensure rows are in the same order
    X_train = X_train.loc[y_train.index, :]
    X_test = X_test.loc[y_test.index, :]

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print(y_train.value_counts(), y_test.value_counts())

    # Train the model with an imbalanced dataset
    if args.bal == 'n':
        if args.fs == 'y':  # Run feature selection
            print('Running feature selection...')
            selected_features = feature_selection_clf(X_train, y_train,
                                                      args.start, args.stop, args.step, args.save, args.prefix,
                                                      args.write, args.type)

            for features in selected_features:
                print(
                    f'Training model with the top {len(features)} features...')
                X_train_fs = X_train.loc[:, features]
                X_test_fs = X_test.loc[:, features]

                if args.alg == 'xgboost':
                    start = time.time()
                    results_cv, results_test = run_xgb(X_train_fs, y_train,
                                                       X_test, y_test, args.y_name, int(
                                                           args.fold), int(args.n),
                                                       f'{args.prefix}_top_{len(features)}', args.ht, args.plot)
                    run_time = time.time() - start
                    save_xgb_results(results_cv, results_test, args,
                                     f'{args.tag}_top_{len(features)}', run_time,
                                     len(X_train_fs), len(features))

                if args.alg == 'autogluon':
                    run_autogluon(X_train_fs, X_test_fs, y_train, y_test,
                                  args.y_name, args.save, f'{args.prefix}_top_{len(features)}')

        else:  # No feature selection
            if args.alg == 'xgboost':
                start = time.time()
                results_cv, results_test = run_xgb(X_train, y_train, X_test, y_test,
                                                   args.y_name, int(args.fold), int(
                                                       args.n), args.prefix, args.ht,
                                                   args.plot)
                run_time = time.time() - start
                save_xgb_results(results_cv, results_test, args, args.tag,
                                 run_time, len(X_train), len(X_train.columns))

            if args.alg == 'autogluon':
                run_autogluon(X_train, X_test, y_train, y_test, args.y_name,
                              args.save, args.prefix)

    # Train the model with a balanced training dataset
    if args.bal == 'y':
        print('Balancing the training set...')
        X_train.insert(0, args.y_name, y_train[X_train.index])
        balanced_datasets = create_balanced(
            X_train, int(args.n_bal), args.y_name)

        for b in range(int(args.n_bal)):
            X_train_bal = balanced_datasets[b].drop(columns=args.y_name)
            y_train_bal = y_train[X_train_bal.index]

            if args.fs == 'y':  # Run feature selection
                print('Running feature selection...')
                selected_features = feature_selection_clf(X_train_bal,
                                                          y_train_bal, args.start, args.stop, args.step, args.save,
                                                          f'{args.prefix}_balanced_{b}', args.write, args.type)

                for features in selected_features:
                    print(
                        f'Training model with the top {len(features)} features...')
                    X_train_bal_fs = X_train_bal.loc[:, features]
                    X_test_fs = X_test.loc[:, features]

                    if args.alg == 'xgboost':
                        start = time.time()
                        results_cv, results_test = run_xgb(X_train_bal_fs,
                                                           y_train_bal, X_test_fs, y_test, args.y_name,
                                                           int(args.fold), int(
                                                               args.n),
                                                           f'{args.prefix}_balanced_{b}_top_{len(features)}',
                                                           args.ht, args.plot)
                        run_time = time.time() - start
                        save_xgb_results(results_cv, results_test, args,
                                         f'{args.tag}_balanced_{b}_top_{len(features)}',
                                         run_time, len(X_train_bal_fs), len(features))

                    if args.alg == 'autogluon':
                        run_autogluon(X_train_bal_fs, X_test_fs, y_train_bal,
                                      y_test, args.y_name, args.save,
                                      f'{args.prefix}_balanced_{b}_top_{len(features)}')

            else:  # No feature selection
                if args.alg == 'xgboost':
                    start = time.time()
                    results_cv, results_test = run_xgb(X_train_bal, y_train_bal,
                                                       X_test, y_test, args.y_name, int(
                                                           args.fold), int(args.n),
                                                       f'{args.prefix}_balanced_{b}', args.ht, args.plot)
                    run_time = time.time() - start
                    save_xgb_results(results_cv, results_test, args,
                                     f'{args.tag}_balanced_{b}', run_time, len(
                                         X_train_bal),
                                     len(X_train_bal.columns))

                if args.alg == 'autogluon':
                    run_autogluon(X_train_bal, X_test, y_train_bal, y_test,
                                  args.y_name, args.save, f'{args.prefix}_balanced_{b}')
