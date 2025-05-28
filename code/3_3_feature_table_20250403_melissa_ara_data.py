#!/usr/bin/env python3

import os
import joblib
import warnings
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error

"""Integrate and impute missing values from 20250403_melissa_ara_data feature
tables."""

os.chdir("/home/seguraab/ara-kinase-prediction/")


def find_best_k(X_train, mask_fraction=0.3, n_splits=10):
    '''Find the best k for KNN imputer by cross-validation. Artifically mask
    feature values, train the imputer, and compare how well the imputed values
    match the original values.

    For binary feature values: an MSE <= 0.01 is considered good.
    For categorical feature values: an MSE of 1-10 is good if values are around 100.
    For continuous feature values: an MSE < 0.01 is good if values are around 0-1.
    '''
    #
    k_values = range(2, 21)  # Define the range of k values to test
    best_score = 1  # Initialize the best score
    best_k = None  # Initialize the best k value
    np.random.seed(2805)  # Set random seed for reproducibility
    #
    # Drop columns with any missing values or infinite values
    X_train_sub = X_train.replace([np.inf, -np.inf], np.nan).copy(deep=True)
    X_train_sub = X_train_sub.dropna(axis=1, how='any')
    if X_train_sub.shape[1] == 0:
        warnings.warn(
            "No columns left after dropping those with missing values.")
        return None, None
    #
    # Loop through each k value and compute the cross-validation score
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=2805)
    for k in k_values:
        fold_errors = []
        #
        for train_idx, val_idx in kf.split(X_train_sub):
            # Split the data into training and validation sets
            X_train_fold = X_train_sub.iloc[train_idx, :]
            X_val_fold = X_train_sub.iloc[val_idx, :]  # keep track of og vals
            #
            # Randomly simulate missingness
            val_masked = X_val_fold.copy(deep=True)  # masked validation feats
            masked_idx = defaultdict(set)  # keep track of masked indices
            idx = [(row, col) for row in range(val_masked.shape[0])
                   for col in range(val_masked.shape[1])]  # row, col indices
            np.random.shuffle(idx)  # Shuffle the indices
            idx_to_mask = int(round(
                mask_fraction * len(idx)))  # number of indices to mask
            for row, col in idx:
                if len(masked_idx[row]) < val_masked.shape[1] - 1:
                    val_masked.iloc[row, col] = np.nan  # mask the values
                    idx_to_mask -= 1
                    masked_idx[row].add(col)  # keep track of masked indices
                    if idx_to_mask == 0:
                        break
            #
            # Instantiate the KNN imputer and fit it to the training data
            imputer = KNNImputer(n_neighbors=k, weights="distance")
            mod = imputer.fit(X_train_fold)
            imputed_val = mod.transform(val_masked)
            #
            # Compute imputation error
            mse = mean_squared_error(imputed_val, X_val_fold.values)
            fold_errors.append(mse)  # store the error for this fold
            #
        score = np.mean(fold_errors)  # average error across folds
        if score < best_score:
            best_score = score
            best_k = k  # update the best k value
    #
    return best_k, best_score


def impute_missing(X_train, X_test, k):
    '''Train a K-Nearest Neighbors imputer with X_train and apply it to data
    X_test and X_train. Drop columns with > 30% missing values. These will not
    be imputed. Return the imputed dataframes and the trained imputer model.'''
    print(f"Shape of training data: {X_train.shape}")
    #
    # Drop columns with > 30% missing values
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_train = X_train.dropna(axis=1, thresh=X_train.shape[0] * 0.7)
    X_test = X_test.loc[:, X_train.columns]  # keep same columns as train
    print(
        f"Shape after dropping columns with > 30% missing values: {X_train.shape}")
    #
    imputer = KNNImputer(n_neighbors=k, weights="distance")
    mod = imputer.fit(X_train)  # train the imputer on the training data
    #
    imputed_train = mod.transform(X_train)
    imputed_test = mod.transform(X_test)
    #
    imputed_train = pd.DataFrame(
        imputed_train, index=X_train.index, columns=X_train.columns)
    imputed_test = pd.DataFrame(
        imputed_test, index=X_test.index, columns=X_test.columns)
    #
    return imputed_train, imputed_test, mod


############## Determine the best k and impute the missing values ##############
# Read file containing labels []
labels_df = pd.read_csv(
    "data/20250403_melissa_ara_data/corrected_data/binary_labels_from_linear_model.csv")

# Load the non-imputed features table
feat_files = sorted([f for f in os.listdir(
    "data/20250403_melissa_ara_data/features") if (f.startswith("20250403_")) and
    (not f.endswith(".csv")) and (f != "20250403_melissa_ara_features_for_binary_clf_pfam.txt")])

# Concatenate the feature files into a single DataFrame
dfs = [pd.read_csv(os.path.join("data/20250403_melissa_ara_data/features", f),
                   sep="\t").set_index(["gene1", "gene2"]) for f in feat_files]
X = pd.concat(dfs, axis=1, ignore_index=False)
del dfs
X.shape  # (138, 5316)

######## Determine the feature types (binary, categorical, continuous) #########
# By inspecting the feature engineering checklist:
binary_categorical_types = ["binary", "number_in_pair", "number_overlapping",
                            "percent_overlapping", "percent_in_pair",
                            "protein_domain_total"]
binary_categorical_cols = [col for col in X.columns if any(
    ftype in col for ftype in binary_categorical_types)]

'''Features ending in _log, _reciprocal, or have retention_rate are continuous
pd.set_option("display.max_rows", 324)
X[binary_categorical_cols].describe().T
'''
# By inspecting the min/max/iqr stats:
actually_continuous_types = ["_log", "retention_rate", "_reciprocal",
                             "lethality_score", "AraNet_percent_overlapping_noTF",
                             "AraNet_percent_overlapping_squared",
                             "protein_protein_interactions"]
binary_categorical_cols = [col for col in binary_categorical_cols if not any(
    ftype in col for ftype in actually_continuous_types)]

# Convert binary and categorical columns to int
X[binary_categorical_cols] = X[binary_categorical_cols].replace(
    [np.inf, -np.inf], np.nan)
NA_mask = X[binary_categorical_cols].isna()
X[binary_categorical_cols] = X[binary_categorical_cols].fillna(0)
X[binary_categorical_cols] = X[binary_categorical_cols].astype(int)
X[binary_categorical_cols] = X[binary_categorical_cols].where(
    ~NA_mask, other=np.nan)  # restore NAs

# Separate the binary from the categorical features; get continuous features
stats = X[binary_categorical_cols].describe().T
binary_cols = stats[(stats["min"] == 0) & (stats["max"] == 1)].index.tolist()
categorical_cols = [
    col for col in binary_categorical_cols if col not in binary_cols]
continuous_cols = [
    col for col in X.columns if col not in binary_categorical_cols]

# save feature types to a file
feature_types = pd.DataFrame(
    [{"feature": feature, "type": "binary"} for feature in binary_cols] +
    [{"feature": feature, "type": "categorical"} for feature in categorical_cols] +
    [{"feature": feature, "type": "continuous"} for feature in continuous_cols])
feature_types.to_csv(
    "data/20250403_melissa_ara_data/features/feature_types_20250403_melissa_ara_features_for_binary_clf.csv",
    index=False, header=True)

############## Determine the best k and impute the missing values ##############
'''Artificially mask feature values and train imputers based on feature type
(binary, categorical, continuous) to determine the best k values for each
feature type. I don't want to make many imputated feature tables for each label
since I want to be able to compare model performance across labels.'''

# Stratified train-test split based on binary_combined_p05 label
labels_df = labels_df.set_index(["gene1", "gene2"])
y_train, y_test = train_test_split(labels_df, test_size=1/11, random_state=2805,
                                   stratify=labels_df["binary_combined_p05"])
X_train = X.loc[y_train.index, :].copy(deep=True)
X_test = X.loc[y_test.index, :].copy(deep=True)

labels_df.insert(1, "dataset", labels_df.apply(
    lambda x: "test" if x.name in y_test.index else "train", axis=1))
labels_df.insert(0, "ID", labels_df.apply(
    lambda x: "_".join(x.name), axis=1))  # create ID column
labels_df.to_csv(
    "data/20250403_melissa_ara_data/corrected_data/binary_labels_from_linear_model_split.csv",
    index=False, header=True)  # save the train-test split
test_instances = labels_df[labels_df["dataset"] == "test"].reset_index()
test_instances["ID"].to_csv(
    "data/20250403_melissa_ara_data/corrected_data/binary_labels_from_linear_model_test_instances.csv",
    index=False, header=False)  # save the test instances

# Determing the best k for imputing each feature type
type_list = ["binary", "categorical", "continuous"]
for i, col_list in enumerate([binary_cols, categorical_cols, continuous_cols]):
    # Find the best k for KNN imputer by cross-validation
    if (X_train[col_list].isna().sum().sum() != 0) &
            (X_test[col_list].isna().sum().sum() != 0):
        best_k, best_score=find_best_k(X_train[col_list])
        print(
            f"Best k for {type_list[i]} features: {best_k} with MSE {best_score}")

# Best k for categorical features: 16 with MSE 0.14037063422138132
# Best k for continuous features: 8 with MSE 0.009429490297723456

# Impute the training and test sets
X_train_categorical, X_test_categorical, mod_categorical=impute_missing(
    X_train[categorical_cols], X_test[categorical_cols], 16)
# Shape of training data: (129, 57)
# Shape after dropping columns with > 30% missing values: (129, 52)
X_train_continuous, X_test_continuous, mod_continuous=impute_missing(
    X_train[continuous_cols], X_test[continuous_cols], 8)
# Shape of training data: (129, 5073)
# Shape after dropping columns with > 30% missing values: (129, 2107)

# Save the imputed feature table to a file
X_train_imputed=pd.concat([X_train[binary_cols], X_train_categorical,
                             X_train_continuous], axis=1, ignore_index=False)
X_test_imputed=pd.concat([X_test[binary_cols], X_test_categorical,
                            X_test_continuous], axis=1, ignore_index=False)
out = pd.concat([X_train_imputed, X_test_imputed], axis=0)
out.insert(0, "ID", out.apply(lambda x: "_".join(x.name), axis=1))
out.to_csv(
    "data/20250403_melissa_ara_data/features/Imputed_20250403_melissa_ara_features_for_binary_clf.csv",
    index=False, header=True)

# Save the imputers to files
joblib.dump(mod_categorical,
            "data/20250403_melissa_ara_data/features/Imputer_categorical_20250403_melissa_ara_features_for_binary_clf.pkl")
joblib.dump(mod_continuous,
            "data/20250403_melissa_ara_data/features/Imputer_continuous_20250403_melissa_ara_features_for_binary_clf.pkl")
