#!/usr/bin/env python3

import os, json, joblib, warnings
import datatable as dt
import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.impute import KNNImputer

""" Integrate feature tables for Dataset_4.txt and impute missing values """
feat_files = os.listdir("/home/seguraab/ara-kinase-prediction/data/2021_cusack_data/Dataset_4_Features")
df_list = []
for f in feat_files:
	# Load the feature table
	tmp = dt.fread("/home/seguraab/ara-kinase-prediction/data/2021_cusack_data/Dataset_4_Features/" + f).to_pandas()
	
	# Set the gene pair identifiers as the index
	tmp['pair_ID'] = tmp['gene1'] + '_' + tmp['gene2']
	tmp.set_index('pair_ID', inplace=True)
	tmp.drop(columns=['gene1', 'gene2'], inplace=True)
	
	df_list.append(tmp) # Append to list

# Concatenate all the feature tables into a single dataframe
dat4_features = pd.concat(df_list, axis=1, ignore_index=False)
dat4_features.shape # (10300, 5340)

# Add the Class label from Dataset_4.txt
dat4 = dt.fread("/home/seguraab/ara-kinase-prediction/data/2021_cusack_data/Dataset_4.txt").to_pandas()
sum(dat4_features.index == dat4["pair_ID"]) # SANITY CHECK: 10300 matches
dat4_features.insert(0, "Class", dat4['Class'].values) # Add the Class column to the features
dat4_features.Class.value_counts()

# Determine what additional features are in dat4_features compared to dat4
len(set(dat4_features.columns) - set(dat4.columns[1:])) # 5174 features in dat4_features and not in dat4
len(set(dat4.columns[1:]) - set(dat4_features.columns)) # 31 features in dat4 and not in dat4_features

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

### Determine if the feature values for the common features are the same
dat4_common = dat4_features.loc[:,dat4_features.columns.isin(dat4.columns)]
dat4_common.shape # (10300, 154)

# How many columns have missing values?
dat4_common.loc[:, dat4_common.isna().sum() != 0].shape # (10300, 108)

# Impute these columns by feature type (dataframes are in df_list)
def find_best_k(dat4_features, feat_df_list, feat_files, y_name="Class", test="test"):
	""" Determine what is the optimal k-value for imputing individual feature
	types using KNNImputer. """
	
	# Drop features with >= 30% missing data
	dat4_feat_to_impute = dat4_features.loc[:, dat4_features.isna().sum()/dat4_features.shape[0] < 0.3]
	print(f"Keeping {dat4_feat_to_impute.shape[1]} out of {dat4_features.shape[1]} with < 30% missing data")
	print(f"Dropping an additional {dat4_feat_to_impute.select_dtypes('number').loc[:,np.isinf(dat4_feat_to_impute.select_dtypes('number')).sum() != 0].shape[1]} columns with infinite values")
	
	# Drop features with only infinite values
	dat4_feat_to_impute = dat4_feat_to_impute.select_dtypes("number").loc\
			[:,np.isinf(dat4_feat_to_impute.select_dtypes("number")).sum() == 0]
	print(f"Imputing {dat4_feat_to_impute.shape[1]} out of {dat4_features.shape[1]} features")
	
	print("Separate the testing data from the training data")
	idx = dat4_features.loc[dat4_feat_to_impute.index, y_name].eq("test")
	test_idx = idx[idx].index
	X_train = dat4_feat_to_impute.drop(index=test_idx, axis=0) # training gene pairs only
	X_test = dat4_feat_to_impute.loc[test_idx] # testing gene pairs only
	y_train = dat4_features.loc[X_train.index, y_name] # Class labels for training gene pairs
	
	# Impute subsets of features by feature type
	imputed_col_names = {}
	best_k_res = {} # store the best k values
	for i,df_feat in enumerate(feat_df_list):
		feat_type = feat_files[i] # feature type
		
		# get the numeric columms to impute
		cols_to_impute = list(set(dat4_feat_to_impute.columns) & set(df_feat.columns))
		imputed_col_names[feat_type] = df_feat[cols_to_impute].select_dtypes("number").columns.tolist() # Store which columns were imputed for each feature type
		print(f"Imputing {len(imputed_col_names[feat_type])} out of {df_feat.shape[1]} features from {feat_type}")
		'''Output:
		Imputing 2 out of 4 features from Dataset_4_features_hydroxylation.txt
		Imputing 2 out of 4 features from Dataset_4_features_formylation.txt
		Imputing 2 out of 4 features from Dataset_4_features_myristoylation.txt
		Imputing 15 out of 15 features from Dataset_4_features_pfam_properties.txt
		Imputing 2 out of 4 features from Dataset_4_features_propionylation.txt
		Imputing 2 out of 4 features from Dataset_4_features_deamination.txt
		Imputing 15 out of 15 features from Dataset_4_features_pfam.txt
		Imputing 1125 out of 1580 features from Dataset_4_features_evolutionary_properties.txt
		Imputing 228 out of 450 features from Dataset_4_features_gene_expression.txt
		Imputing 1216 out of 2432 features from Dataset_4_features_functional_annotations.txt
		Imputing 25 out of 25 features from Dataset_4_features_aa_len.txt
		Imputing 26 out of 205 features from Dataset_4_features_network_properties.txt
		Imputing 25 out of 25 features from Dataset_4_features_iso_pt.txt
		Imputing 2 out of 4 features from Dataset_4_features_oxidation.txt
		Imputing 2 out of 4 features from Dataset_4_features_acetylation.txt
		Imputing 284 out of 565 features from Dataset_4_features_epigenetics.txt
		'''
		
		# Test out different n_neighbors values
		best_score = -1
		best_k = 0
		for k in range(3, 20):
			imputer = KNNImputer(n_neighbors=k, weights="distance", add_indicator=True)
			with warnings.catch_warnings():
				warnings.simplefilter("ignore", category=RuntimeWarning)  # Ignorar todas las advertencias
				mod = make_pipeline(imputer, MinMaxScaler(), LogisticRegression(max_iter=1000))
			
			scores = cross_val_score(mod,
									 X_train[cols_to_impute],
									 y_train.squeeze(),
									 cv=5,error_score='raise').mean()
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
	
	imputed_train_feat = [] # imputed training features
	imputed_test_feat = [] # imputed testing features
	for feat_type, cols_to_impute in imputed_col_names.items():
		dat4_feat_to_impute = dat4_features.loc[:,cols_to_impute]
		print(f"Imputing {dat4_feat_to_impute.shape[1]} out of {dat4_features.shape[1]} features from {feat_type} with {best_k_res[feat_type]['best_k']} neighbors")
		'''Output:
		Imputing 2 out of 5341 features from Dataset_4_features_hydroxylation.txt with 3 neighbors
		Imputing 2 out of 5341 features from Dataset_4_features_formylation.txt with 3 neighbors
		Imputing 2 out of 5341 features from Dataset_4_features_myristoylation.txt with 3 neighbors
		Imputing 30 out of 5341 features from Dataset_4_features_pfam_properties.txt with 3 neighbors
		Imputing 2 out of 5341 features from Dataset_4_features_propionylation.txt with 3 neighbors
		Imputing 2 out of 5341 features from Dataset_4_features_deamination.txt with 3 neighbors
		Imputing 30 out of 5341 features from Dataset_4_features_pfam.txt with 3 neighbors
		Imputing 1125 out of 5341 features from Dataset_4_features_evolutionary_properties.txt with 5 neighbors
		Imputing 228 out of 5341 features from Dataset_4_features_gene_expression.txt with 3 neighbors
		Imputing 1216 out of 5341 features from Dataset_4_features_functional_annotations.txt with 3 neighbors
		Imputing 25 out of 5341 features from Dataset_4_features_aa_len.txt with 3 neighbors
		Imputing 26 out of 5341 features from Dataset_4_features_network_properties.txt with 14 neighbors
		Imputing 25 out of 5341 features from Dataset_4_features_iso_pt.txt with 3 neighbors
		Imputing 2 out of 5341 features from Dataset_4_features_oxidation.txt with 3 neighbors
		Imputing 2 out of 5341 features from Dataset_4_features_acetylation.txt with 3 neighbors
		Imputing 284 out of 5341 features from Dataset_4_features_epigenetics.txt with 10 neighbors
		'''
		
		# Train-test split
		idx = dat4_features.loc[dat4_feat_to_impute.index, y_name].eq("test")
		test_idx = idx[idx].index
		X_train = dat4_feat_to_impute.drop(index=test_idx, axis=0) # training gene pairs only
		X_test = dat4_feat_to_impute.loc[test_idx] # testing gene pairs only
	
		print(f"Imputing features from {feat_type} in dat4_features")
		# make sure bool columns are now int
		# X_train = X_train.astype({col: 'int' for col in X_train.columns if X_train[col].dtypes == 'bool'})
		X_train = X_train.astype({col: 'int' for col in X_train.select_dtypes(include=['bool']).columns})
		
		# Train imputer on training set
		imp_train, mod = impute_missing(X_train, best_k_res[feat_type]['best_k'])
		imputed_train_feat.append(imp_train)
		joblib.dump(mod, f"/home/seguraab/ara-kinase-prediction/data/2021_cusack_data/Dataset_4_Features/Imputer_{feat_type.replace('.txt','')}_train.joblib")
		
		# Impute the test set using the trained imputer
		imp_test = mod.transform(X_test)
		imp_test = pd.DataFrame(imp_test, index=X_test.index, columns=X_test.columns)
		imputed_test_feat.append(imp_test)
		
		del X_train, X_test, imp_train, imp_test, mod
	
	# Combine all imputed features into a single dataframe
	X_train_imp = pd.concat(imputed_train_feat, axis=1, ignore_index=False)
	X_test_imp = pd.concat(imputed_test_feat, axis=1, ignore_index=False)
	X_imp = pd.concat([X_train_imp, X_test_imp], axis=0) # (10300, 3003) # Total number of features after imputation
	
	# Insert the class label
	X_imp.insert(0, "Class", dat4_features.loc[X_imp.index, "Class"].values)
	
	# Save the feature table
	X_imp.to_csv("/home/seguraab/ara-kinase-prediction/data/2021_cusack_data/Dataset_4_Features/Imputed_Dataset_4_final_table.csv")
	X_imp.shape # (10300, 3004)


# Run the imputation process
imputed_col_names, best_k_res = find_best_k(dat4_features, df_list, feat_files)
run_imputation(imputed_col_names, best_k_res, dat4_features)

""" Calculate the correlation between features from the original Dataset_4.txt
and the regenerated features in Imputed_Dataset_4_final_table.csv """