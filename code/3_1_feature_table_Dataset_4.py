#!/usr/bin/env python3

import os
import json
import joblib
import warnings
import datatable as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.impute import KNNImputer

""" Integrate feature tables for Dataset_4.txt and impute missing values """
feat_files = os.listdir(
    "/home/seguraab/ara-kinase-prediction/data/2021_cusack_data/Dataset_4_Features")
df_list = []
actual_feat_files = []
for f in feat_files:
    if (f.startswith("Dataset_4_features_")) & (f != "Dataset_4_features_evolutionary_properties.txt"):
        # Load the feature table
        tmp = dt.fread(
            "/home/seguraab/ara-kinase-prediction/data/2021_cusack_data/Dataset_4_Features/" + f).to_pandas()

        # Set the gene pair identifiers as the index
        tmp['pair_ID'] = tmp['gene1'] + '_' + tmp['gene2']
        tmp.set_index('pair_ID', inplace=True)
        tmp.drop(columns=['gene1', 'gene2'], inplace=True)

        if f == "Dataset_4_features_functional_annotations.txt":
            # assume gene pairs have no pathway relationship, we don't want to impute these
            tmp.fillna(value=0, inplace=True)

        if f == "Dataset_4_features_pfam_properties.txt":
            continue  # skip this file, it was not meant to be included
        df_list.append(tmp)  # Append to list
        actual_feat_files.append(f)  # Append to list of actual feature files

# Concatenate all the feature tables into a single dataframe
dat4_features = pd.concat(df_list, axis=1, ignore_index=False)
# concatenating results in duplicate columns
dat4_features = dat4_features.loc[:, ~dat4_features.columns.duplicated()]
print(dat4_features.shape)  # (10300, 5366)

# Add the Class label from Dataset_4.txt
dat4 = dt.fread(
    "/home/seguraab/ara-kinase-prediction/data/2021_cusack_data/Dataset_4.txt").to_pandas()
# SANITY CHECK: 10300 matches
print(sum(dat4_features.index == dat4["pair_ID"]))
# Add the Class column to the features
dat4_features.insert(0, "Class", dat4['Class'].values)
dat4_features.Class.value_counts()
dat4_features.loc[dat4_features.Class == "test"].index.to_frame().\
    to_csv("/home/seguraab/ara-kinase-prediction/data/2021_cusack_data/Dataset_4_test_instances.txt",
           index=False, header=False)

# Determine what additional features are in dat4_features compared to dat4
# 5223 features in dat4_features and not in dat4
print(len(set(dat4_features.columns) - set(dat4.columns[1:])))
# 39 features in dat4 and not in dat4_features
print(len(set(dat4.columns[1:]) - set(dat4_features.columns)))

# The features that are not in dat4_features. I need to make them.
pprint(set(dat4.columns[1:]) - set(dat4_features.columns))
"""
The following missing features are from evolutionary properties and functional
annotations (GOSLIM). I will need to generate these at some point, but for the
purpose of figuring out what's wrong with the classification models, I won't
include them for now. What we can do is exclude them from Dataset_4.txt so that 
we can compare the regenerated features to Cusack 2021's Dataset_4.txt features.

{'binary_A_WGD',
 'continuous_GOSLIM_developmental_processes_number_in_pair_reciprocal',
 'continuous_GOSLIM_nucleus_number_in_pair_log',
 'continuous_GOSLIM_number_overlapping_squared',
 'continuous_GOSLIM_other_biological_processes_number_in_pair_log',
 'continuous_GOSLIM_other_cytoplasmic_components_number_in_pair_squared',
 'continuous_GOSLIM_percent_overlapping_log',
 'continuous_GOSLIM_plasma_membrane_number_in_pair_noTF',
 'continuous_GOSLIM_protein_binding_number_in_pair_reciprocal',
 'continuous_GOSLIM_response_to_abiotic_or_biotic_stimulus_number_in_pair_squared',
 'continuous_GOSLIM_signal_transduction_number_in_pair_log',
 'continuous_GOSLIM_total_squared',
 'continuous_GOSLIM_transcription_DNA_dependent_number_in_pair_log',
 'continuous_GOSLIM_transferase_activity_number_in_pair_log',
 'continuous_amino_acid_similarity_log',
 'continuous_ka_average_log',
 'continuous_ka_difference_noTF',
 'continuous_ka_ks_average_noTF',
 'continuous_ka_ks_difference_log',
 'continuous_ka_ks_max_reciprocal',
 'continuous_ka_ks_min_reciprocal',
 'continuous_ka_ks_pair_total_squared',
 'continuous_ka_max_noTF',
 'continuous_ka_min_noTF',
 'continuous_ka_pair_total_log',
 'continuous_ks_average_squared',
 'continuous_ks_difference_reciprocal',
 'continuous_ks_max_reciprocal',
 'continuous_ks_min_noTF',
 'continuous_ks_pair_total_log',
 'continuous_nucleotide_similarity_squared'} """

# Determine if the feature values for the common features are the same
dat4_common = dat4_features.loc[:, dat4_features.columns.isin(dat4.columns)]
print(dat4_common.shape)  # (10300, 144)

# How many columns have missing values?
print(dat4_common.loc[:, dat4_common.isna().sum() != 0].shape)  # (10300, 106)

# Impute these columns by feature type (dataframes are in df_list)


def find_best_k(dat4_features, feat_df_list, feat_files, y_name="Class", test="test"):
    """ Determine what is the optimal k-value for imputing individual feature
    types using KNNImputer. """

    # Drop features with >= 30% missing data
    dat4_feat_to_impute = dat4_features.loc[:, dat4_features.isna(
    ).sum()/dat4_features.shape[0] < 0.3]
    print(
        f"Keeping {dat4_feat_to_impute.shape[1]} out of {dat4_features.shape[1]} with < 30% missing data")
    print(
        f"Dropping an additional {dat4_feat_to_impute.select_dtypes('number').loc[:,np.isinf(dat4_feat_to_impute.select_dtypes('number')).sum() != 0].shape[1]} columns with infinite values")

    # Drop features with only infinite values
    dat4_feat_to_impute = dat4_feat_to_impute.select_dtypes(
        "number").loc[:, np.isinf(dat4_feat_to_impute.select_dtypes("number")).sum() == 0]
    print(
        f"Imputing {dat4_feat_to_impute.shape[1]} out of {dat4_features.shape[1]} features")

    print("Separate the testing data from the training data")
    idx = dat4_features.loc[dat4_feat_to_impute.index, y_name].eq("test")
    test_idx = idx[idx].index
    X_train = dat4_feat_to_impute.drop(
        index=test_idx, axis=0)  # training gene pairs only
    # X_train.to_csv(
    #    "/home/seguraab/ara-kinase-prediction/data/2021_cusack_data/Dataset_4_Features/Imputation_Dataset_4_X_train_table.csv")
    X_test = dat4_feat_to_impute.loc[test_idx]  # testing gene pairs only
    # Class labels for training gene pairs
    y_train = dat4_features.loc[X_train.index, y_name]
    # y_train.to_csv(
    #     "/home/seguraab/ara-kinase-prediction/data/2021_cusack_data/Dataset_4_Features/Imputation_Dataset_4_y_train_table.csv",
    #     header=True)

    # Impute subsets of features by feature type
    imputed_col_names = {}
    best_k_res = {}  # store the best k values

    for i, df_feat in enumerate(feat_df_list):
        feat_type = feat_files[i]  # feature type

        # get the numeric columms to impute
        cols_to_impute = list(
            set(dat4_feat_to_impute.columns) & set(df_feat.columns))
        imputed_col_names[feat_type] = df_feat[cols_to_impute].select_dtypes(
            "number").columns.tolist()  # Store which columns were imputed for each feature type
        if len(imputed_col_names[feat_type]) > 0:
            print(
                f"Imputing {len(imputed_col_names[feat_type])} out of {df_feat.shape[1]} features from {feat_type}")
            if feat_type == "Dataset_4_features_functional_annotations.txt":
                print(
                    f"Will not impute {len(imputed_col_names[feat_type])} features from {feat_type} since they are all 0")
                print(X_train[cols_to_impute].isna().sum() / 161)
                X_train.drop(
                    columns=imputed_col_names[feat_type], inplace=True)
                X_test.drop(columns=imputed_col_names[feat_type], inplace=True)
                continue
        else:
            print(f"No features to impute from {feat_type}")
            continue

        '''Output:
		Keeping 4435 out of 5367 with < 30% missing data
		Dropping an additional 608 columns with infinite values
		Imputing 3588 out of 5367 features
		Separate the testing data from the training data
		Imputing 30 out of 30 features from Dataset_4_features_evolutionary_properties_Lethality score.txt
		Imputing 2 out of 4 features from Dataset_4_features_hydroxylation.txt
		Imputing 12 out of 30 features from Dataset_4_features_evolutionary_properties_Lethality binary.txt
		Imputing 2 out of 4 features from Dataset_4_features_formylation.txt
		Imputing 2 out of 4 features from Dataset_4_features_myristoylation.txt
		Imputing 2 out of 4 features from Dataset_4_features_propionylation.txt
		Imputing 261 out of 500 features from Dataset_4_features_evolutionary_properties_Ka_Ks.txt
		Imputing 2 out of 4 features from Dataset_4_features_deamination.txt
		Imputing 15 out of 15 features from Dataset_4_features_pfam.txt
		No features to impute from Dataset_4_features_evolutionary_properties_Reciprocal best match.txt
		Imputing 344 out of 500 features from Dataset_4_features_evolutionary_properties_Ks.txt
		Imputing 228 out of 450 features from Dataset_4_features_gene_expression.txt
		Imputing 1824 out of 2432 features from Dataset_4_features_functional_annotations.txt
		Will not impute 1824 features from Dataset_4_features_functional_annotations.txt since they are all 0
		Imputing 28 out of 30 features from Dataset_4_features_evolutionary_properties_Gene family size.txt
		Imputing 22 out of 30 features from Dataset_4_features_evolutionary_properties_Retention rate.txt
		Imputing 25 out of 25 features from Dataset_4_features_aa_len.txt
		Imputing 26 out of 205 features from Dataset_4_features_network_properties.txt
		Imputing 450 out of 500 features from Dataset_4_features_evolutionary_properties_Ka.txt
		Imputing 25 out of 25 features from Dataset_4_features_iso_pt.txt
		Imputing 2 out of 4 features from Dataset_4_features_oxidation.txt
		Imputing 2 out of 4 features from Dataset_4_features_acetylation.txt
		Imputing 284 out of 565 features from Dataset_4_features_epigenetics.txt
		'''

        # Test out different n_neighbors values
        best_score = -1
        best_k = 0
        for k in range(3, 20):
            imputer = KNNImputer(
                n_neighbors=k, weights="distance", add_indicator=True)
            with warnings.catch_warnings():
                # Ignorar todas las advertencias
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mod = make_pipeline(imputer, MinMaxScaler(),
                                    LogisticRegression(max_iter=1000))

            scores = cross_val_score(mod,
                                     X_train[cols_to_impute],
                                     y_train.squeeze(),
                                     cv=5, error_score='raise').mean()
            if scores > best_score:
                best_score = scores
                best_k = k

        best_k_res[feat_type] = {'best_k': best_k, 'best_score': best_score}

    # Save results
    pd.DataFrame.from_dict(best_k_res, orient="index").to_csv(
        "/home/seguraab/ara-kinase-prediction/data/2021_cusack_data/Dataset_4_Features/Imputation_best_k_values.csv",
        index=True)
    with open(
            "/home/seguraab/ara-kinase-prediction/data/2021_cusack_data/Dataset_4_Features/Imputed_features.json", "w") as outf:
        json.dump(imputed_col_names, outf, indent=4)

    return imputed_col_names, best_k_res


def impute_missing(X, k):
    '''Instantiate a K-Nearest Neighbors imputer and apply it to data X'''
    imputer = KNNImputer(n_neighbors=k, weights="distance")
    mod = imputer.fit(X)
    imputed = mod.transform(X)
    imputed = pd.DataFrame(imputed, index=X.index, columns=X.columns)
    return imputed, mod


def run_imputation(imputed_col_names, best_k_res, dat4_features, y_name="Class", test="test"):
    '''Impute the missing values using KNNImputer.'''

    imputed_train_feat = []  # imputed training features
    imputed_test_feat = []  # imputed testing features
    functional_annotations_features = pd.DataFrame()
    for feat_type, cols_to_impute in imputed_col_names.items():
        dat4_feat_to_impute = dat4_features.loc[:, cols_to_impute]
        if dat4_feat_to_impute.shape[1] > 0:
            if feat_type == "Dataset_4_features_functional_annotations.txt":
                print(
                    f"Will not impute {dat4_feat_to_impute.shape[1]} features from {feat_type} since they are all 0")
                functional_annotations_features = dat4_features.loc[:,
                                                                    cols_to_impute]
            else:
                print(
                    f"Imputing {dat4_feat_to_impute.shape[1]} out of {dat4_features.shape[1]} features from {feat_type} with {best_k_res[feat_type]['best_k']} neighbors")
        else:
            print(f"No features to impute from {feat_type}")
            continue

        '''Output:
		Imputing 30 out of 5367 features from Dataset_4_features_evolutionary_properties_Lethality score.txt with 3 neighbors
		Imputing features from Dataset_4_features_evolutionary_properties_Lethality score.txt in dat4_features
		Imputing 2 out of 5367 features from Dataset_4_features_hydroxylation.txt with 3 neighbors
		Imputing features from Dataset_4_features_hydroxylation.txt in dat4_features
		Imputing 12 out of 5367 features from Dataset_4_features_evolutionary_properties_Lethality binary.txt with 3 neighbors
		Imputing features from Dataset_4_features_evolutionary_properties_Lethality binary.txt in dat4_features
		Imputing 2 out of 5367 features from Dataset_4_features_formylation.txt with 3 neighbors
		Imputing features from Dataset_4_features_formylation.txt in dat4_features
		Imputing 2 out of 5367 features from Dataset_4_features_myristoylation.txt with 3 neighbors
		Imputing features from Dataset_4_features_myristoylation.txt in dat4_features
		Imputing 2 out of 5367 features from Dataset_4_features_propionylation.txt with 3 neighbors
		Imputing features from Dataset_4_features_propionylation.txt in dat4_features
		Imputing 261 out of 5367 features from Dataset_4_features_evolutionary_properties_Ka_Ks.txt with 5 neighbors
		Imputing features from Dataset_4_features_evolutionary_properties_Ka_Ks.txt in dat4_features
		Imputing 2 out of 5367 features from Dataset_4_features_deamination.txt with 3 neighbors
		Imputing features from Dataset_4_features_deamination.txt in dat4_features
		Imputing 15 out of 5367 features from Dataset_4_features_pfam.txt with 3 neighbors
		Imputing features from Dataset_4_features_pfam.txt in dat4_features
		No features to impute from Dataset_4_features_evolutionary_properties_Reciprocal best match.txt
		Imputing 344 out of 5367 features from Dataset_4_features_evolutionary_properties_Ks.txt with 4 neighbors
		Imputing features from Dataset_4_features_evolutionary_properties_Ks.txt in dat4_features
		Imputing 228 out of 5367 features from Dataset_4_features_gene_expression.txt with 3 neighbors
		Imputing features from Dataset_4_features_gene_expression.txt in dat4_features
		Will not impute 1824 features from Dataset_4_features_functional_annotations.txt since they are all 0
		Imputing 28 out of 5367 features from Dataset_4_features_evolutionary_properties_Gene family size.txt with 3 neighbors
		Imputing features from Dataset_4_features_evolutionary_properties_Gene family size.txt in dat4_features
		Imputing 22 out of 5367 features from Dataset_4_features_evolutionary_properties_Retention rate.txt with 3 neighbors
		Imputing features from Dataset_4_features_evolutionary_properties_Retention rate.txt in dat4_features
		Imputing 25 out of 5367 features from Dataset_4_features_aa_len.txt with 3 neighbors
		Imputing features from Dataset_4_features_aa_len.txt in dat4_features
		Imputing 26 out of 5367 features from Dataset_4_features_network_properties.txt with 14 neighbors
		Imputing features from Dataset_4_features_network_properties.txt in dat4_features
		Imputing 450 out of 5367 features from Dataset_4_features_evolutionary_properties_Ka.txt with 4 neighbors
		Imputing features from Dataset_4_features_evolutionary_properties_Ka.txt in dat4_features
		Imputing 25 out of 5367 features from Dataset_4_features_iso_pt.txt with 3 neighbors
		Imputing features from Dataset_4_features_iso_pt.txt in dat4_features
		Imputing 2 out of 5367 features from Dataset_4_features_oxidation.txt with 3 neighbors
		Imputing features from Dataset_4_features_oxidation.txt in dat4_features
		Imputing 2 out of 5367 features from Dataset_4_features_acetylation.txt with 3 neighbors
		Imputing features from Dataset_4_features_acetylation.txt in dat4_features
		Imputing 284 out of 5367 features from Dataset_4_features_epigenetics.txt with 10 neighbors
		Imputing features from Dataset_4_features_epigenetics.txt in dat4_features
		Imputed features shape: (10300, 1764)
		'''

        # Train-test split
        idx = dat4_features.loc[dat4_feat_to_impute.index, y_name].eq("test")
        test_idx = idx[idx].index
        X_train = dat4_feat_to_impute.drop(
            index=test_idx, axis=0)  # training gene pairs only
        X_test = dat4_feat_to_impute.loc[test_idx]  # testing gene pairs only

        print(f"Imputing features from {feat_type} in dat4_features")
        # make sure bool columns are now int
        X_train = X_train.astype(
            {col: 'int' for col in X_train.select_dtypes(include=['bool']).columns})

        if feat_type == "Dataset_4_features_functional_annotations.txt":
            continue

        # Train imputer on training set
        imp_train, mod = impute_missing(
            X_train, best_k_res[feat_type]['best_k'])
        imputed_train_feat.append(imp_train)
        joblib.dump(
            mod, f"/home/seguraab/ara-kinase-prediction/data/2021_cusack_data/Dataset_4_Features/Imputer_{feat_type.replace('.txt','')}_train.joblib")

        # Impute the test set using the trained imputer
        imp_test = mod.transform(X_test)
        imp_test = pd.DataFrame(
            imp_test, index=X_test.index, columns=X_test.columns)
        imputed_test_feat.append(imp_test)

        del X_train, X_test, imp_train, imp_test, mod

    # Combine all imputed features into a single dataframe
    X_train_imp = pd.concat(imputed_train_feat, axis=1, ignore_index=False)
    X_test_imp = pd.concat(imputed_test_feat, axis=1, ignore_index=False)
    # (10300, 2995) # Total number of features after imputation
    X_imp = pd.concat([X_train_imp, X_test_imp], axis=0)
    print(f"Imputed features shape: {X_imp.shape}")

    # Combine with the functional annotations features
    X_imp = pd.concat([X_imp, functional_annotations_features],
                      axis=1, ignore_index=False)
    print(
        f"Imputed features shape after adding functional annotations: {X_imp.shape}")

    # Insert the class label
    X_imp.insert(0, "Class", dat4_features.loc[X_imp.index, "Class"].values)

    # Save the feature table
    X_imp.to_csv(
        "/home/seguraab/ara-kinase-prediction/data/2021_cusack_data/Dataset_4_Features/Imputed_Dataset_4_final_table.csv")
    print(X_imp.shape)  # (10300, 2996)


# Run the imputation process
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="invalid value encountered in reduce")
imputed_col_names, best_k_res = find_best_k(
    dat4_features, df_list, actual_feat_files)
run_imputation(imputed_col_names, best_k_res, dat4_features)

""" Calculate the correlation between features from the original Dataset_4.txt
and the regenerated features in Imputed_Dataset_4_final_table.csv """

og_dat4 = dt.fread(
    "/home/seguraab/ara-kinase-prediction/data/2021_cusack_data/Dataset_4.txt").to_pandas()
new_dat4 = dt.fread(
    "/home/seguraab/ara-kinase-prediction/data/2021_cusack_data/Dataset_4_Features/Imputed_Dataset_4_final_table.csv").to_pandas()
og_dat4.set_index("pair_ID", inplace=True)
new_dat4.set_index("pair_ID", inplace=True)

# Save the list of feature names, so that I can re-use the trained model to infer
# the label for the TAIR10 kinase gene pairs.
new_dat4.columns[1:].to_frame().to_csv(
    "/home/seguraab/ara-kinase-prediction/data/2021_cusack_data/Dataset_4_Features/Imputed_Dataset_4_final_table_features_list.txt",
    index=False, header=False)

# Correlation between columns within a dataframe
palette = sns.color_palette("pastel", 3)
color_map = og_dat4.Class.map(dict(zip(og_dat4.Class.unique(), palette)))
color_map2 = new_dat4.Class.map(dict(zip(new_dat4.Class.unique(), palette)))
savedir = "/home/seguraab/ara-kinase-prediction/data/2021_cusack_data/"

sns.clustermap(og_dat4.drop(columns="Class").corr(), cmap="RdBu_r",
               cbar_kws={"label": "Correlation"}, method="average", center=0,
               xticklabels=False, yticklabels=False)
plt.tight_layout()
plt.savefig(savedir + "Corr_features_Dataset_4.png")
plt.close()

sns.heatmap(new_dat4.drop(columns="Class").corr(), cmap="RdBu_r",
            cbar_kws={"label": "Correlation"}, center=0,
            xticklabels=False, yticklabels=False)
plt.tight_layout()
plt.savefig(savedir + "Dataset_4_Features/Corr_features_imputed_Dataset_4_v2.png")
plt.close()
# Note: added _v2 since I had to re-run the evolutionary properties features code, all the other features are the same as before.

sns.heatmap(og_dat4.drop(columns="Class").T.corr(), cmap="RdBu_r",
            cbar_kws={"label": "Correlation"}, center=0,
            xticklabels=False, yticklabels=False)
plt.tight_layout()
plt.savefig(savedir + "Corr_instances_Dataset_4.png")
plt.close()

sns.heatmap(new_dat4.drop(columns="Class").T.corr(), cmap="RdBu_r",
            cbar_kws={"label": "Correlation"}, center=0,
            xticklabels=False, yticklabels=False)
plt.tight_layout()
plt.savefig(
    savedir + "Dataset_4_Features/Corr_instances_imputed_Dataset_4_v2.png")
plt.close()
# Note: there was a typo, so the original file was actually still og_dat4. I still added _v2, but this is technically
# the "first" version of the imputed features.

# Note: I did not re-run anything below this line with the new imputed features,
# since technically only the evolutionary properties features were changed.
# These have different names in the dataframes so they weren't compared.

# Correlations between the original and imputed features
feat_corrs = {}
for feat in og_dat4.columns[1:]:
    if feat in new_dat4.columns:
        feat_corrs[feat] = og_dat4[feat].corr(
            new_dat4.loc[og_dat4.index, feat])  # Calculate the correlation

pd.DataFrame.from_dict(feat_corrs, orient="index", columns=["Correlation"]).\
    plot.hist(bins=30, edgecolor='black', alpha=0.7)
plt.title("Correlation between original and imputed features")
plt.savefig(savedir + "Dataset_4_Features/Corr_features_histogram_imputed.png")
# conclusion: common features are different... but the imputation will have an effect.

pd.DataFrame.from_dict(feat_corrs, orient="index", columns=["Correlation"]).\
    to_csv(savedir + "Dataset_4_Features/Corr_features_histogram_imputed.csv")

# Correlation between the original and re-generated, but not imputed, features
files = [f for f in os.listdir(
    savedir + "Dataset_4_Features/") if f.startswith("Dataset_4")]
df_list = []
for f in files:
    tmp = dt.fread(savedir + "Dataset_4_Features/" + f).to_pandas()
    tmp['pair_ID'] = tmp['gene1'] + '_' + tmp['gene2']
    tmp.set_index('pair_ID', inplace=True)
    tmp.drop(columns=['gene1', 'gene2'], inplace=True)
    df_list.append(tmp)

new_dat4_not_imputed = pd.concat(df_list, axis=1, ignore_index=False)

feat_corrs = {}
for feat in og_dat4.columns[1:]:
    if feat in new_dat4_not_imputed.columns:
        try:
            feat_corrs[feat] = og_dat4[feat].corr(
                new_dat4_not_imputed.loc[og_dat4.index, feat].iloc[:, 0])  # Calculate the correlation
        except:
            feat_corrs[feat] = og_dat4[feat].corr(
                new_dat4_not_imputed.loc[og_dat4.index, feat])  # Calculate the correlation

pd.DataFrame.from_dict(feat_corrs, orient="index", columns=["Correlation"]).\
    plot.hist(bins=30, edgecolor='black', alpha=0.7)
plt.title("Correlation between original and re-generated features (not imputed yet)")
plt.savefig(
    savedir + "Dataset_4_Features/Corr_features_histogram_not_imputed.png")
plt.close()
# conclusion: most have a PCC=1, but there are are several that don't. Will need
# to see the csv file to see which ones are not correlated.

pd.DataFrame.from_dict(feat_corrs, orient="index", columns=["Correlation"]).\
    to_csv(savedir + "Dataset_4_Features/Corr_features_histogram_not_imputed.csv")
