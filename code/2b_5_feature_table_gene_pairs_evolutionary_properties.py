#!/usr/bin/env python
'''Create the evolutionary properties feature table for single genes as instances.'''

import os, swifter, concurrent
import numpy as np
import pandas as pd
import datatable as dt
import networkx as nx
from tqdm import tqdm

__author__ = 'Kenia Segura Abá'

def dict_val_counts(dictionary, val_idx=None):
	'''Similar to pandas value_counts() method. This function counts the number 
	of keys with the same value in a dictionary. The function does not work on 
	nested dictionaries.
	
	Parameters:
		- dictionary (dict): A dictionary with keys and values (float, int, str, or list)
		- val_idx (int): The index of the element of interest if the value is a list
	'''

	inverted_dict = {}
	for key, value in dictionary.items():
		if (not isinstance(value, list)) and (value not in inverted_dict.keys()): # value is not a list
			inverted_dict[value] = [key]
		elif (isinstance(value, list)) and (value[val_idx] not in inverted_dict.keys()): # value is a list
			inverted_dict[value[val_idx]] = [key]
		else:
			if val_idx == None:
				inverted_dict[value].append(key)
			else:
				inverted_dict[value[val_idx]].append(key)
	
	# Count the number of keys for each value
	dict_counts = {value: len(keys) for value, keys in inverted_dict.items()}
	print(dict_counts)


def calc_binary_adj_feat_values(adj, col_name):
	'''Extract the binary feature values for gene pairs in the `genes` file 
	from an adjacency matrix'''
	
	G = nx.from_pandas_adjacency(adj)
	
	features = []
	for i in tqdm(range(len(genes))):
		gene1 = genes.gene1[i]
		gene2 = genes.gene2[i]
		
		# Calculate feature values
		try:
			# get the edge value
			edge = [e for e in list(G.edges()) \
					if (gene1 in e[0] or gene1 in e[1]) \
					and (gene2 in e[0] or gene2 in e[1])]
			if len(edge) == 0:
				features.append([gene1, gene2, 0])
			else:
				features.append([gene1, gene2, 1])
		
		except nx.exception.NetworkXError:
			# At least one gene identifier is not in the network
			features.append([gene1, gene2, np.nan])
	
	col_names=['gene1', 'gene2', col_name]
	
	return pd.DataFrame(features, columns=col_names)


def apply_transformations(values, method):
	''' Apply transformations to a dataframe of continuous feature values.
	
	Args:
		- df (pd.DataFrame): A data frame of feature values
		- chklst_df (pd.DataFrame): Feature engineering checklist
	
	Returns an expanded dataframe with additional transformations on continuous columns:
		- Binned values
		- Logarithm
		- Reciprocal
		- Squared
	'''
	
	# Apply the transformations to individual columns
	if method == 'Binary':
		return values
	
	if method == 'Continuous':
		if values.name.endswith('_binned'):
			# Bin the feature value
			new_values = pd.qcut(values, q=4, labels=False, duplicates='drop')
		elif values.name.endswith('_log'):
			# Calculate the log
			new_values = values.apply(lambda x: np.log10(x) if x > 0 else np.nan)
		elif values.name.endswith('_reciprocal'):
			# Calculate the reciprocal
			new_values = values.apply(lambda x: 1/x if x != 0 else np.nan)
		elif values.name.endswith('_squared'):
			# Calculate the square
			new_values = values ** 2
		elif values.name.endswith('_noTF'):
			# No transformation applied
			return values
		
		return new_values


def calc_feature_values(gene1_values, gene2_values, genes, gene_values, calc_type):
	"""Calculate feature values for a gene pair.
	Args:
		gene1_values (nested dictionary): genes as outermost keys, column names in gene_values as innermost keys
		gene2_values (nested dictionary): genes as outermost keys, column names in gene_values as innermost keys
		genes (pandas.DataFrame): dataframe with two columns ("gene1" and "gene2") of gene IDs. Rows represent a gene pair.
		gene_values (pandas.DataFrame): feature values from which feature values will be calculated from for a gene pair
		calc_type (str): the type of calculation to generate the feature values
	Returns:
		nested dictionary: gene pair tuples as outermost keys, column names in gene_values as innermost keys
	"""
	
	continuous_values = {}
	for col in gene_values.columns:
		new_col = col + '_' + calc_type.lower().replace(' ', '_')
		
		for pair in tqdm(genes.iterrows(), total=len(genes), desc=f"Processing gene pairs for {new_col}"):
			try:
				value1 = gene1_values[pair[1].gene1][col]
				value2 = gene2_values[pair[1].gene2][col]
			except KeyError: # one or both of the genes have no feature data
				continuous_values[(pair[1].gene1, pair[1].gene2)] = {}
				continuous_values[(pair[1].gene1, pair[1].gene2)][new_col] = None
				continue
			
			# # take the median across splice variants of the gene
			# if type(value1) != np.float64: 
			# 	value1 = gene_values.T.filter(like=key).median(axis=1)
			# 	value1.name = key
			# if type(value2) != np.float64:
			# 	value2 = gene_values.T.filter(like=key2).median(axis=1)
			# 	value2.name = key2
			
			# calculate the continuous value for a gene pair
			if (pair[1].gene1, pair[1].gene2) not in continuous_values:
				continuous_values[(pair[1].gene1, pair[1].gene2)] = {}
			
			if calc_type in ['Number in pair', 'Pair total']:
				continuous_values[(pair[1].gene1, pair[1].gene2)][new_col] = value1 + value2
			elif calc_type == 'Average':
				continuous_values[(pair[1].gene1, pair[1].gene2)][new_col] = (value1 + value2) / 2
			elif calc_type == 'Difference':
				continuous_values[(pair[1].gene1, pair[1].gene2)][new_col] = abs(value1 - value2)
			elif calc_type == 'Maximum':
				try:
					continuous_values[(pair[1].gene1, pair[1].gene2)][new_col] = max(value1, value2)
				except ValueError: # if value1 and value2 are pandas Series
					continuous_values[(pair[1].gene1, pair[1].gene2)][new_col] = np.maximum(value1, value2)
			elif calc_type == 'Minimum':
				try:
					continuous_values[(pair[1].gene1, pair[1].gene2)][new_col] = min(value1, value2)
				except ValueError:
					continuous_values[(pair[1].gene1, pair[1].gene2)][new_col] = np.minimum(value1, value2)
			elif calc_type == '% similarity':
				continuous_values[(pair[1].gene1, pair[1].gene2)][new_col] = None
	
	return continuous_values


def apply_transformations_df(df):
	''' Apply transformations to a dataframe of continuous feature values.
	
	Args:
		- df (pd.DataFrame): A data frame of feature values
	
	Returns an expanded dataframe with additional transformations on continuous columns:
		- Binned values
		- Logarithm
		- Reciprocal
		- Squared
	'''
	df_list = []
	df_reset = df.reset_index()
	
	# Apply the transformations to individual columns
	for column in df.columns:
		if column not in df.index.names:
			# Bin the feature value
			bins = pd.qcut(df_reset[column], q=4, labels=False, duplicates='drop')
			bins.name = column + '_binned'
			
			# Calculate the log
			log_vals = df_reset[column].swifter.apply(lambda x: np.log10(x) if x > 0 else np.nan)
			log_vals.name = column + '_log'
			
			# Calculate the reciprocal
			reciprocal = df_reset[column].swifter.apply(lambda x: 1/x if x != 0 else np.nan)
			reciprocal.name = column + '_reciprocal'
			
			# Calculate the square
			square = df_reset[column] ** 2
			square.name = column + '_squared'
			
			df_list.append(pd.concat([df_reset[column], bins, log_vals,
				reciprocal, square], axis=1, ignore_index=False))
			
	return pd.concat(df_list, axis=1, ignore_index=False)


def calc_paml_feat_values(col_list):
	'''Create the paml feature table for gene pairs in the `genes` file.'''
	
	cols = [col for col in paml_feat.loc[:, col_list].columns \
			if not ('_binned' in col or '_log' in col \
			or '_reciprocal' in col or '_squared' in col)] # columns to calculate feature values for
	
	feature_values = paml_feat[cols].T.to_dict() # genes as keys
	gene1_values = {gene: feature_values[gene] for gene in genes.gene1.values if gene in feature_values}
	gene2_values = {gene: feature_values[gene] for gene in genes.gene2.values if gene in feature_values}
	
	cols_features = []
	for calc_type in ['Average', 'Difference', 'Maximum', 'Minimum', 'Pair total']:
		print('Applying the calculation:', calc_type)
		
		cols_values = calc_feature_values(gene1_values, gene2_values,
			genes, paml_feat[cols], calc_type) # calculate the feature values
		
		cols_features.append(pd.DataFrame.from_dict(cols_values, orient="index"))
		del cols_values
	
	# Combine features into a single dataframe
	cols_features = pd.concat(cols_features, axis=1, ignore_index=False)
	cols_features.index = pd.MultiIndex.from_tuples(list(cols_features.index), names=["gene1", "gene2"])
	
	# Apply transformations
	cols_features = apply_transformations_df(cols_features)
	
	return cols_features


if __name__ == '__main__':
	os.chdir('/home/seguraab/ara-kinase-prediction/')

	# Feature checklist file
	checklist = pd.read_csv('data/Features/02_evolutionary_properties_feature_list.csv')

	# Open the single gene feature tables
	feat1 = pd.read_csv('data/Features/evolutionary_properties_single_genes_feature_table_part1.csv', index_col=0)
	paml_feat = dt.fread('data/Features/evolutionary_properties_single_genes_feature_table_part2_paml_only.csv').to_pandas()
	blast_hits = dt.fread('data/Features/evolutionary_properties_single_genes_reciprocal_best_match_adjacency_matrix.csv.gz').to_pandas()
	blast_hits.set_index('gene', inplace=True)
	blast_hits = blast_hits.astype(int) # convert boolean to binary
	
	# Calculate the median dN, dS, and dN/dS values across splice variants of a gene
	paml_feat["gene"] = paml_feat["C0"].str.split('.').str[0] # get gene ID from mRNA ID
	paml_feat = paml_feat.drop(columns="C0").groupby("gene").median()
	
	# If any of the splice variants for two genes were reciprocal blast hits, set their value to 1
	sum(blast_hits.columns==blast_hits.index) # row and col gene names are the same
	row_groups = blast_hits.index.str.split('.').str[0]
	col_groups = blast_hits.columns.str.split('.').str[0]
	blast_hits_grouped = (blast_hits.groupby(row_groups, axis=0).any().\
		groupby(col_groups, axis=1).any()).astype(int)
	blast_dict = blast_hits_grouped.to_dict()
 
	# Instances file
	# genes = pd.read_csv('data/instances_dataset_1.txt', sep='\t')
	# genes = pd.read_csv('data/2021_cusack_data/Dataset_4.txt', sep='\t')
	genes = pd.read_csv('data/Kinase_genes/instances_tair10_kinases.txt', sep='\t')
	# genes = genes["pair_ID"].str.split("_", expand=True)
	# genes.columns = ['gene1', 'gene2']
	
	###################### Calculate the features values #######################
	features = {}

	print("First make the paml features...")
	ka_ks_cols = paml_feat.columns.str.contains('dN/dS')
	ks_cols = paml_feat.columns.str.contains('_dS_')
	ka_cols = paml_feat.columns.str.contains('_dN_')
	features['Ka/Ks'] = calc_paml_feat_values(ka_ks_cols)
	features['Ks'] = calc_paml_feat_values(ks_cols)
	features['Ka'] = calc_paml_feat_values(ka_cols)

	# print("Make the reciprocal best match feature...")
	# features['Reciprocal best match'] = calc_binary_adj_feat_values(blast_hits_grouped, "binary_reciprocal_best_match")
	# features['Reciprocal best match'].set_index(['gene1', 'gene2'], inplace=True)

	# Make the rest of the features
	feature_values = feat1.T.to_dict() # genes as keys
	gene1_values = {gene: feature_values[gene] for gene in genes.gene1.values if gene in feature_values}
	gene2_values = {gene: feature_values[gene] for gene in genes.gene2.values if gene in feature_values}

	ml_features = checklist.groupby(
		['NEW Feature name', 'Calculation for gene pair', 'Data processing method'])\
		['ML model feature name'].apply(list).reset_index()
	
	for idx, row in tqdm(ml_features.iterrows()):
		ml_feat_names = row['ML model feature name']
		feat_name = row['NEW Feature name']
		calc_type = row['Calculation for gene pair']
		feat_type = row['Data processing method']
		print(f'Generating features for {feat_name}')
		
		if feat_name in ['0 blast res', 'tbd', 'arabidopsis paml combined results']:
			continue
		
		elif feat_name in ['gene family size', 'retention rate']:
			target_col = '_'.join([feat_type.lower(), feat_name.replace(' ', '_')])
			values = calc_feature_values(gene1_values, gene2_values,
				genes, pd.DataFrame(feat1[target_col]), calc_type)
			
		elif (feat_name == 'lethality') & (feat_type == 'Continuous'):
			values = calc_feature_values(gene1_values, gene2_values,
				genes, pd.DataFrame(feat1['continuous_lethality_score']), calc_type)
			
		elif (feat_name == 'lethality') & (feat_type == 'Binary'):
			values = calc_feature_values(gene1_values, gene2_values,
				genes, pd.DataFrame(feat1['binary_lethality']), calc_type)
			
		values = pd.DataFrame.from_dict(values, orient="index") #type: ignore
		values = pd.concat([values] * len(ml_feat_names), axis=1)
		values.columns = ml_feat_names
		values.index = pd.MultiIndex.from_tuples(list(values.index), names=["gene1", "gene2"]) #type: ignore
		values = apply_transformations_df(values)
		features[f"{feat_name}_{calc_type}"] = values
		del values
	
	# Save the feature table to a file!
	# pd.concat(features.values(), axis=1, ignore_index=False).to_csv(
	# 	'data/Features/evolutionary_properties_gene_pairs_features.txt', sep='\t')
	# pd.concat(features.values(), axis=1, ignore_index=False).to_csv(
	# 	'data/2021_cusack_data/Dataset_4_Features/Dataset_4_features_evolutionary_properties.txt',
	# 	sep='\t')
	pd.concat(features.values(), axis=1, ignore_index=False).to_csv(
		'data/Kinase_genes/features/TAIR10_kinases_features_evolutionary_properties.txt',
		sep='\t', chunksize=100)
	
	############### Update the evolutionary properties checklist ###############
	

	'''############################################################################
	# Check how many of our kinase gene pair instances have single gene lethality data
	uniq_genes = set(genes.gene1.unique().tolist() + genes.gene2.unique().tolist())

	num_with_data = []
	for gene in uniq_genes:
		if gene in pheno.keys():
			num_with_data.append(gene)

	len(num_with_data)

	# Check how many of our kinase gene pairs have gene interaction info on Biogrid
	gi = pd.read_csv('data/BIOGRID_data/arabidopsis_gi_biogrid.txt', sep='\t', header=None)
	pair_sets = {tuple(sorted([gene1, gene2])) for gene1, gene2 in zip(genes.gene1, genes.gene2)}
	gi_sets = {tuple(sorted([gene1, gene2])) for gene1, gene2 in zip(gi[5], gi[6])}
	len(gi_sets) # 294 unique gene pairs from BIOGRID
	
	gi_sets_with_dups = [tuple(sorted([gene1, gene2])) for gene1, gene2 in zip(gi[5], gi[6])]
	
	from collections import Counter
	dict_val_counts(Counter(gi_sets_with_dups))
	pair_sets.intersection(gi_sets) # only 7 gene pairs
	'''
