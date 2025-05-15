#!/usr/bin/env python3

import os
import gc
import joblib
import warnings
import datatable as dt
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

"""Integrate and impute missing values from the TAIR10 kinase feature tables."""

os.chdir("/home/seguraab/ara-kinase-prediction/")

############################### Impute features ###############################
# Imputed Dataset_4.txt features list
imputed_dat4_features = pd.read_csv(
    "data/2021_cusack_data/Dataset_4_Features/Imputed_Dataset_4_final_table_features_list.txt",
    header=None)
"""These are the features I used to train RF and XGB models with, thus I need to
impute the same ones for the TAIR10 kinase feature tables. Even though, there are
probably some that will have >= 30% missing data. I guess I should have taken the
common set of features instead, trained the models, then infer the label of this
TAIR10 kinase dataset."""

missing_data = {}  # amount of missing data in features that will be imputed
feat_tables = {}  # imputed feature tables
feat_files = os.listdir("data/Kinase_genes/features")

for i, f in enumerate(feat_files):
    if f.startswith("TAIR10_kinases_features_"):
        print(f"Processing {f}...")
    else:
        continue

    if f == "TAIR10_kinases_features_functional_annotations.txt":
        # continue # only uncomment if trying to get the missing data for the other files
        '''I commented this out since I already ran it once and got what I needed'''
        # with open(f"data/Kinase_genes/features/Imputed_{f.replace('.txt', '_by_fillna.txt')}", "w") as out:
        # 	# process the file in chunks
        # 	chunk_iter = pd.read_csv(f"data/Kinase_genes/features/{f}", sep="\t", chunksize=100_000)
        # 	for i, chunk in enumerate(chunk_iter):
        # 		chunk.fillna(value=0, inplace=True) # assume gene pairs have no pathway relationship
        # 		chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        # 		chunk.to_csv(out, sep="\t", index=False, header=(i==0)) # only write header on first chunk

        # # Missing data
        # df = dt.fread(f"data/Kinase_genes/features/{f}") # converting to pandas kills the process
        # df = df[:, ["gene1", "gene2"] + [col for col in df.names if col in imputed_dat4_features[0].values]]
        # df.replace([np.inf, -np.inf], np.nan)
        # col_missing = df.countna().to_pandas().T / df.shape[0] * 100
        # missing_data[f] = col_missing.round(1).sort_values(0, ascending=False)
        # del df
        continue

    # read in the feature table
    df = dt.fread(f"data/Kinase_genes/features/{f}").to_pandas()
    df.set_index(["gene1", "gene2"], inplace=True)  # set index to gene pairs
    print(f"Shape of {f}: {df.shape}")

    if f == "TAIR10_kinases_features_pfam.txt":
        continue  # skip this file

    # convert binary columns to int
    binary_cols = df.select_dtypes(include=["bool"]).columns
    for col in binary_cols:
        df[col] = df[col].astype(int)

    # keep only the features that coincide with the imputed Dataset_4.txt features
    df = df.loc[:, df.columns.isin(imputed_dat4_features[0].values)]
    print(
        f"Shape after keeping only the features that coincide with the imputed Dataset_4.txt features: {df.shape}")

    # estimate the amount of missing data column-wise
    # col_missing = df.isnull().sum().sort_values(ascending=False) / df.shape[0] * 100
    # missing_data[f] = col_missing.round(1)

    # Impute missing values using the KNNImputer trained on the imputed Dataset_4.txt features
    if df.shape[1] == 0:
        print(
            f"Skipping {f} since it has no features in common with the imputed Dataset_4.txt features")
        del df
        continue

    # check if the imputer was trained on this feature table
    if f not in feat_tables.keys():
        mod_name = f.replace("TAIR10_kinases_features_",
                             "").replace(".txt", "")
        if mod_name == "pfam_properties":
            mod_name = "pfam"
            '''I have to check why the pfam_properties imputer has two
            propionylation features instead of the pfam properties columns.'''

        mod = joblib.load(
            f"data/2021_cusack_data/Dataset_4_Features/Imputer_Dataset_4_features_{mod_name}_train.joblib")

        # features need to be in the same order as the imputer was trained on
        df_imputed = mod.transform(df.loc[:, mod.feature_names_in_])
        '''I'm getting the same warning as before, which I still don't know how to fix.
        /home/seguraab/miniconda3/envs/py310/lib/python3.10/site-packages/numpy/core/fromnumeric.py:88: RuntimeWarning: invalid value encountered in reduce
        return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
        '''
        # Write imputed table to file
        pd.DataFrame(df_imputed, index=df.index, columns=mod.feature_names_in_).\
            to_csv(
            f"data/Kinase_genes/features/Imputed_{f}", index=False, header=True, sep="\t")

        # Save imputed features to the dictionary, these will be integrated later
        feat_tables[f] = df_imputed
        del df, df_imputed, mod

# Save missing data to file
missing_data_df = pd.concat([mdf for mdf in missing_data.values()], axis=0)
missing_data_df.to_csv(
    "data/Kinase_genes/features/TAIR10_kinases_features_percent_missing_data.csv")


################# Integrate imputed features and infer labels ##################
# Open all file handles for chunked reading
imputed_files = sorted([f for f in os.listdir(
    "data/Kinase_genes/features") if f.startswith("Imputed_")])
chunksize = 50000
readers = [pd.read_csv(os.path.join("data/Kinase_genes/features", f), sep="\t",
                       chunksize=chunksize) for f in imputed_files]

# Load the pre-trained model
rf_updated_mod = joblib.load(
    "data/2021_cusack_data/Dataset_4_Features/output_clf/RF/rf_clf_imputed_Dataset_4_updated_models.pkl")
y_preds = {"Class": [], "Probability": []}  # predicted labels

# Get the gene pairs
idx_df = pd.read_csv(
    "data/Kinase_genes/features/TAIR10_kinases_features_aa_len.txt", sep="\t", header=0)
idx_df = idx_df[["gene1", "gene2"]]

with open("data/Kinase_genes/features/Imputed_TAIR10_kinases_final_table.csv", "w") as out:
    chunk_idx = 0
    while chunk_idx < 48:  # 48 chunks (about 2.4M rows)
        print(f"Processing chunk {chunk_idx} out of 47")
        chunk_list = []
        # Get the next chunk from each file
        try:
            for i, reader in enumerate(readers):
                chunk = next(reader)
                chunk_list.append(chunk)
        except StopIteration:
            print(f"File {imputed_files[i]} has no more data")
            break
        #
        print(f"Concatenate chunks along columns")
        integrated_chunk = pd.concat(chunk_list, axis=1)
        integrated_chunk.index = pd.MultiIndex.from_frame(
            idx_df.iloc[integrated_chunk.index, :])
        #
        # Write to output file
        integrated_chunk.to_csv(out, header=(chunk_idx == 0), index=True)
        #
        print("Infer the labels using pre-trained model")
        chunk_y_pred_updated = rf_updated_mod.predict(
            integrated_chunk.loc[:, rf_updated_mod.feature_names_in_])
        chunk_y_pred_updated_proba = rf_updated_mod.predict_proba(
            integrated_chunk.loc[:, rf_updated_mod.feature_names_in_])
        y_preds["Class"].extend(chunk_y_pred_updated)
        y_preds["Probability"].extend(chunk_y_pred_updated_proba[:, 1])
        del integrated_chunk, chunk_y_pred_updated, chunk_y_pred_updated_proba
        del chunk_list
        gc.collect()
        chunk_idx += 1

y_preds = pd.DataFrame(y_preds)
y_preds.index = pd.MultiIndex.from_frame(idx_df)
y_preds.to_csv(
    "data/Kinase_genes/features/Imputed_TAIR10_kinases_final_table_y_preds.csv",
    index=True, header=True)


##### Train imputers on the re-made Dataset_4.txt features by feature type #####
# Load the X_train table and y_train label
# These were made in 4c_feature_table_Dataset_4.txt.py
X_train = pd.read_csv(
    "data/2021_cusack_data/Dataset_4_Features/Imputation_Dataset_4_X_train_table.csv",
    index_col=0)
y_train = pd.read_csv(
    "data/2021_cusack_data/Dataset_4_Features/Imputation_Dataset_4_y_train_table.csv",
    index_col=0)

# Determine the feature types (binary, categorical, continuous)
binary_categorical_types = ["binary", "number_in_pair", "number_overlapping",
                            # from feature engineering checklist
                            "percent_overlapping", "protein_domain_total"]
binary_categorical_cols = [col for col in X_train.columns if any(
    ftype in col for ftype in binary_categorical_types)]

'''Features ending in _log, _reciprocal, or have retention_rate are continuous
pd.set_option("display.max_rows", 101)
X_train[binary_categorical_cols].describe().T'''

actually_continuous_types = ["_log", "retention_rate", "_reciprocal"]
binary_categorical_cols = [col for col in binary_cols if not any(
    # binary & categorical features
    ftype in col for ftype in actually_continuous_types)]

# convert binary and categorical columns to int
X_train[binary_categorical_cols] = X_train[binary_categorical_cols].replace(
    [np.inf, -np.inf], np.nan)  # replace infinite values with NaN
NA_mask = X_train[binary_categorical_cols].isnull()  # NaN locations
X_train[binary_categorical_cols] = X_train[binary_categorical_cols].fillna(0)
X_train[binary_categorical_cols] = X_train[binary_categorical_cols].astype(int)

# convert 0 values that should be NaN to NaN
X_train[binary_categorical_cols] = X_train[binary_categorical_cols].where(
    ~NA_mask, other=np.nan)

# separate binary from categorical columns
stats = X_train[binary_categorical_cols].describe().T
binary_cols = stats[(stats['min'] == 0) & (stats['max'] == 1)].index.tolist()
categorical_cols = [
    col for col in binary_categorical_cols if col not in binary_cols]

# determine continuous columns
continuous_cols = [
    col for col in X_train.columns if col not in binary_categorical_cols]

# save feature types to a file
feature_types = pd.DataFrame(
    [{"feature": feature, "type": "binary"} for feature in binary_cols] +
    [{"feature": feature, "type": "categorical"} for feature in categorical_cols] +
    [{"feature": feature, "type": "continuous"} for feature in continuous_cols])
feature_types.to_csv(
    "data/2021_cusack_data/Dataset_4_Features/Imputation_Dataset_4_X_train_feature_types.csv",
    index=False, header=True)

# Train an imputer for each feature type
type_list = ["binary", "categorical", "continuous"]
best_k_res = {"binary": {}, "categorical": {}, "continuous": {}}
imputed_train_feat = []  # list to store imputed feature

for i, col_list in enumerate([binary_cols, categorical_cols, continuous_cols]):
    # Determine the best k for KNNImputer
    best_score = -1
    best_k = 0
    for k in range(3, 20):
        imputer = KNNImputer(
            n_neighbors=k, weights="distance", add_indicator=True)
        with warnings.catch_warnings():  # ignore warnings
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if type_list[i] != "binary":
                mod = make_pipeline(imputer, MinMaxScaler(),
                                    LogisticRegression(max_iter=1000))
            else:
                mod = make_pipeline(imputer, LogisticRegression(max_iter=1000))
        #
        # Cross-validate the model
        scores = cross_val_score(mod, X_train[col_list], y_train.squeeze(),
                                 cv=10, error_score="raise").mean()
        if scores > best_score:
            best_score = scores
            best_k = k  # the best k for this feature type
    print(
        f"Best k for {type_list[i]} features: {best_k} with score: {best_score:.2f}")
    best_k_res[type_list[i]] = {"best_k": best_k, "best_score": best_score}
    #
    # Create the imputer
    from sklearn.impute import KNNImputer
    imputer = KNNImputer(n_neighbors=best_k, weights="distance")
    if type_list[i] != "binary":
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(X_train[col_list])
        mod = imputer.fit(scaled_data)
        imp_train = mod.transform(scaled_data)
    else:
        mod = imputer.fit(X_train[col_list])
        imp_train = mod.transform(X_train[col_list])
    #
    imp_train = pd.DataFrame(imp_train, index=X_train.index,
                             columns=X_train[col_list].columns)
    imputed_train_feat.append(imp_train)
    #
    # Save the imputer to file
    joblib.dump(mod,
                f"data/2021_cusack_data/Dataset_4_Features/Imputer_Dataset_4_X_train_{type_list[i]}_features_model.joblib")
    del mod, imputer, imp_train, scaled_data

# Save the imputed features to a file
imputed_train_feat = pd.concat(imputed_train_feat, axis=1, ignore_index=False)
imputed_train_feat.insert(0, "Class", y_train)
imputed_train_feat.to_csv(
    "data/2021_cusack_data/Dataset_4_Features/Imputed_Dataset_4_X_train_final_table_by_bin_cat_cont_type.csv",
    index=True, header=True)
del imputed_train_feat, X_train, y_train


######### Drop gene pairs with no data before imputing by feature type #########
# Load the non-imputed features table
feat_files = sorted([f for f in os.listdir(
    "data/Kinase_genes/features") if (f.startswith("TAIR10_")) and
    (not f.endswith(".csv")) and (f != "TAIR10_kinases_features_pfam.txt")])

readers = [pd.read_csv(os.path.join("data/Kinase_genes/features", f), sep="\t",
                       chunksize=50000) for f in feat_files]

# Get the gene pairs
idx_df = pd.read_csv(
    "data/Kinase_genes/features/TAIR10_kinases_features_aa_len.txt", sep="\t", header=0)
idx_df = idx_df[["gene1", "gene2"]]

# Get the feature types
feature_types = pd.read_csv(
    "data/2021_cusack_data/Dataset_4_Features/Imputation_Dataset_4_X_train_feature_types.csv",
    index_col=0)
categorical_cols = feature_types[
    feature_types["type"] == "categorical"].index.values.tolist()
continuous_cols = feature_types[
    feature_types["type"] == "continuous"].index.values.tolist()

# Load the imputers
binary_imputer = joblib.load(
    "data/2021_cusack_data/Dataset_4_Features/Imputer_Dataset_4_X_train_binary_features_model.joblib")
categorical_imputer = joblib.load(
    "data/2021_cusack_data/Dataset_4_Features/Imputer_Dataset_4_X_train_categorical_features_model.joblib")
continuous_imputer = joblib.load(
    "data/2021_cusack_data/Dataset_4_Features/Imputer_Dataset_4_X_train_continuous_features_model.joblib")

# Load the pre-trained model
rf_mod = joblib.load(
    "data/2021_cusack_data/Dataset_4_Features/output_clf/RF/rf_clf_imputed_Dataset_4by_bin_cat_cont_type_models.pkl")
rf_fs_mod = joblib.load(
    "data/2021_cusack_data/Dataset_4_Features/output_clf/RF_FS/rf_fs_clf_imputed_Dataset_4by_bin_cat_cont_type_feature_set_above_p0.9.txt_models.pkl")
y_preds = {"Class": {"RF_before_FS": [], "RF_FS": []}, "Probability": {
    "RF_before_FS": [], "RF_FS": []}}  # predicted labels

# Impute the features by feature type in chunks
gene_pairs_kept = []
with open("data/Kinase_genes/features/Imputed_TAIR10_kinases_final_table_by_bin_cat_cont_type.csv", "w") as out:
    chunk_idx = 0
    while chunk_idx < 48:  # 48 chunks (about 2.4M rows)
        print(f"Processing chunk {chunk_idx} out of 47")
        chunk_list = []
        #
        # Get the next chunk from each file
        try:
            for i, reader in enumerate(readers):
                chunk = next(reader)
                chunk_list.append(chunk)
                del chunk  # free memory
        except StopIteration:
            print(f"File {imputed_files[i]} has no more data")
            break
        #
        print(f"Concatenate chunks along columns")
        integrated_chunk = pd.concat(chunk_list, axis=1)
        if "gene1" in integrated_chunk.columns:
            integrated_chunk = integrated_chunk.drop(
                columns=["gene1", "gene2"])
        integrated_chunk.index = pd.MultiIndex.from_frame(
            idx_df.iloc[integrated_chunk.index, :])
        #
        # Drop gene pairs with >= 50% missing data
        integrated_chunk = integrated_chunk.dropna(
            thresh=integrated_chunk.shape[1] * 0.5)
        gene_pairs_kept.extend(
            integrated_chunk.index.values.tolist())
        #
        imputed_features = []
        # Impute the binary features
        '''Note: when I print the integrate_chunk binary features, they're not all
        actually binary, even though for the Dataset_4.txt they were. That's because
        we were looking at only 161 gene pairs, versus here we're looking at thousands'''
        binary_imp = binary_imputer.transform(
            integrated_chunk.loc[:, binary_imputer.feature_names_in_])
        binary_imp = pd.DataFrame(binary_imp, index=integrated_chunk.index,
                                  columns=binary_imputer.feature_names_in_)
        imputed_features.append(binary_imp)
        #
        # Impute the categorical features
        scaler = MinMaxScaler()
        categorical_imp = categorical_imputer.transform(
            scaler.fit_transform(integrated_chunk.loc[:, categorical_cols]))
        categorical_imp = pd.DataFrame(categorical_imp, index=integrated_chunk.index,
                                       columns=categorical_cols)
        imputed_features.append(categorical_imp)
        #
        # Impute the continuous features
        scaler = MinMaxScaler()
        continuous_imp = continuous_imputer.transform(
            scaler.fit_transform(integrated_chunk.loc[:, continuous_cols]))
        continuous_imp = pd.DataFrame(continuous_imp, index=integrated_chunk.index,
                                      columns=continuous_cols)
        #
        # Concatenate the imputed features
        imputed_chunk = pd.concat([binary_imp, categorical_imp, continuous_imp],
                                  axis=1, ignore_index=False)
        #
        # Write the imputed chunk to a file
        imputed_chunk.to_csv(out, header=(chunk_idx == 0), index=True)
        #
        print("Infer the labels using pre-trained model")
        chunk_y_pred = rf_mod.predict(
            imputed_chunk.loc[:, rf_mod.feature_names_in_])  # predicted label
        chunk_y_pred_proba = rf_mod.predict_proba(
            imputed_chunk.loc[:, rf_mod.feature_names_in_])  # probabilities
        chunk_y_pred_fs = rf_fs_mod.predict(
            imputed_chunk.loc[:, rf_fs_mod.feature_names_in_])
        chunk_y_pred_proba_fs = rf_fs_mod.predict_proba(
            imputed_chunk.loc[:, rf_fs_mod.feature_names_in_])

        y_preds["Class"]["RF_before_FS"].extend(chunk_y_pred)
        y_preds["Probability"]["RF_before_FS"].extend(chunk_y_pred_proba[:, 1])
        y_preds["Class"]["RF_FS"].extend(chunk_y_pred_fs)
        y_preds["Probability"]["RF_FS"].extend(chunk_y_pred_proba_fs[:, 1])
        del imputed_chunk, chunk_y_pred, chunk_y_pred_proba, chunk_y_pred_fs,
        del chunk_y_pred_proba_fs
        gc.collect()
        chunk_idx += 1

# I added this code later since I had not put in the rf_fs_mod lines yet.
# I didn't want to hastle with re-running integration or trying to skip lines.
# chunk_iter = pd.read_csv(
#     'data/Kinase_genes/features/Imputed_TAIR10_kinases_final_table_by_bin_cat_cont_type.csv', chunksize=50000)
# gene_pairs_kept = []
# for i, chunk in enumerate(chunk_iter):
#     print(i)
#     gene_pairs_kept.extend(chunk[['gene1', 'gene2']].values.tolist())
#     chunk_y_pred = rf_mod.predict(chunk.loc[:, rf_mod.feature_names_in_])
#     chunk_y_pred_proba = rf_mod.predict_proba(
#         chunk.loc[:, rf_mod.feature_names_in_])
#     chunk_y_pred_fs = rf_fs_mod.predict(
#         chunk.loc[:, rf_fs_mod.feature_names_in_])
#     chunk_y_pred_proba_fs = rf_fs_mod.predict_proba(
#         chunk.loc[:, rf_fs_mod.feature_names_in_])
#     y_preds["Class"]["RF_before_FS"].extend(chunk_y_pred)
#     y_preds["Probability"]["RF_before_FS"].extend(chunk_y_pred_proba[:, 1])
#     y_preds["Class"]["RF_FS"].extend(chunk_y_pred_fs)
#     y_preds["Probability"]["RF_FS"].extend(chunk_y_pred_proba_fs[:, 1])

# Save the predicted labels to a file
y_preds_flat = {f'{outer}_{inner}': value for outer,
                inner_dict in y_preds.items() for inner, value in inner_dict.items()}
y_preds_df = pd.DataFrame(y_preds_flat)
y_preds_df.index = pd.MultiIndex.from_frame(
    pd.DataFrame(gene_pairs_kept), names=["gene1", "gene2"])
y_preds_df.to_csv(
    "data/Kinase_genes/features/Imputed_TAIR10_kinases_final_table_by_bin_cat_cont_type_y_preds.csv",
    index=True, header=True)

# Consolidate which gene pairs were predicted "positive" with higher confidence
y_preds_df.loc[(y_preds_df["Class_RF_before_FS"] == "positive") &
               (y_preds_df["Class_RF_FS"] == "positive")].sort_values(
    by="Probability_RF_FS", ascending=False).to_csv(
    "data/Kinase_genes/features/Imputed_TAIR10_kinases_final_table_by_bin_cat_cont_type_y_preds_positive.csv",
    index=True, header=True)
