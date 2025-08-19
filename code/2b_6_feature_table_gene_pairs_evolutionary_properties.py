#!/usr/bin/env python
'''Create the evolutionary properties feature table for single genes as instances.'''

import os
import numpy as np
import pandas as pd
import datatable as dt
import networkx as nx
from tqdm import tqdm
import dask.dataframe as dd

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
        if (not isinstance(value, list)) and (value not in inverted_dict.keys()):  # value is not a list
            inverted_dict[value] = [key]
        # value is a list
        elif (isinstance(value, list)) and (value[val_idx] not in inverted_dict.keys()):
            inverted_dict[value[val_idx]] = [key]
        else:
            if val_idx == None:
                inverted_dict[value].append(key)
            else:
                inverted_dict[value[val_idx]].append(key)

    # Count the number of keys for each value
    dict_counts = {value: len(keys) for value, keys in inverted_dict.items()}
    print(dict_counts)


# def calc_binary_adj_feat_values(adj, col_name):
# 	'''Extract the binary feature values for gene pairs in the `genes` file
# 	from an adjacency matrix'''

# 	G = nx.from_pandas_adjacency(adj)

# 	features = []
# 	for i in tqdm(range(len(genes))):
# 		gene1 = genes.gene1[i]
# 		gene2 = genes.gene2[i]

# 		# Calculate feature values
# 		try:
# 			# get the edge value
# 			edge = [e for e in list(G.edges()) \
# 					if (gene1 in e[0] or gene1 in e[1]) \
# 					and (gene2 in e[0] or gene2 in e[1])]
# 			if len(edge) == 0:
# 				features.append([gene1, gene2, 0])
# 			else:
# 				features.append([gene1, gene2, 1])

# 		except nx.exception.NetworkXError:
# 			# At least one gene identifier is not in the network
# 			features.append([gene1, gene2, np.nan])

# 	col_names=['gene1', 'gene2', col_name]

# 	return pd.DataFrame(features, columns=col_names)


def calc_binary_adj_feat_values(adj, col_name, genes):
    '''Extract the binary feature values for gene pairs in the `genes` file 
    from an adjacency matrix.'''

    # Create graph from adjacency matrix
    G = nx.from_pandas_adjacency(adj)

    # Unique edges in the graph
    edges_set = set(G.edges())

    # New feature values
    features = []

    # Iterate through the gene pairs
    for gene1, gene2 in tqdm(zip(genes.gene1, genes.gene2), total=len(genes), desc="Processing gene pairs"):
        # Verify if the edge exists in the graph
        if (gene1, gene2) in edges_set or (gene2, gene1) in edges_set:
            features.append([gene1, gene2, 1])  # Edge exists
        else:
            features.append([gene1, gene2, 0])  # No edge

    # Return the feature as a DataFrame
    col_names = ['gene1', 'gene2', col_name]
    return pd.DataFrame(features, columns=col_names).set_index(['gene1', 'gene2'])


# def apply_transformations(values, method):
# 	''' Apply transformations to a dataframe of continuous feature values.

# 	Args:
# 		- df (pd.DataFrame): A data frame of feature values
# 		- chklst_df (pd.DataFrame): Feature engineering checklist

# 	Returns an expanded dataframe with additional transformations on continuous columns:
# 		- Binned values
# 		- Logarithm
# 		- Reciprocal
# 		- Squared
# 	'''

# 	# Apply the transformations to individual columns
# 	if method == 'Binary':
# 		return values

# 	if method == 'Continuous':
# 		if values.name.endswith('_binned'):
# 			# Bin the feature value
# 			new_values = pd.qcut(values, q=4, labels=False, duplicates='drop')
# 		elif values.name.endswith('_log'):
# 			# Calculate the log
# 			new_values = values.apply(lambda x: np.log10(x) if x > 0 else np.nan)
# 		elif values.name.endswith('_reciprocal'):
# 			# Calculate the reciprocal
# 			new_values = values.apply(lambda x: 1/x if x != 0 else np.nan)
# 		elif values.name.endswith('_squared'):
# 			# Calculate the square
# 			new_values = values ** 2
# 		elif values.name.endswith('_noTF'):
# 			# No transformation applied
# 			return values

# 		return new_values


def calc_feature_values(gene1_values_df, gene2_values_df, cols, calc_type):
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

    # Vectorized calculation of feature values
    if calc_type in ['Number in pair', 'Pair total']:
        result = gene1_values_df.values + gene2_values_df.values
    elif calc_type == 'Average':
        result = (gene1_values_df.values + gene2_values_df.values) / 2
    elif calc_type == 'Difference':
        result = abs(gene1_values_df.values - gene2_values_df.values)
    elif calc_type == 'Maximum':
        result = np.maximum(gene1_values_df.values, gene2_values_df.values)
    elif calc_type == 'Minimum':
        result = np.minimum(gene1_values_df.values, gene2_values_df.values)
    else:
        # Default to None for unsupported calc_type
        result = np.full_like(gene1_values_df.values, None)

    # Crear un DataFrame con los resultados
    result_df = pd.DataFrame(result, columns=[
                             f"{col}_{calc_type.lower().replace(' ', '_')}" for col in cols])
    result_df.index = pd.MultiIndex.from_arrays(
        [gene1_values_df.index, gene2_values_df.index], names=["gene1", "gene2"])

    return result_df


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
    # df_list = []
    df_reset = df.reset_index()
    transformed_columns = {}

    # Apply the transformations to individual columns
    for column in tqdm(df.columns, desc="Processing transformations"):
        if column not in df.index.names:
            # No transformation applied
            transformed_columns[column + '_noTF'] = df_reset[column]

            # Bin the feature value
            transformed_columns[column + '_binned'] = pd.qcut(
                df_reset[column], q=4, labels=False, duplicates='drop')

            # Calculate the log
            with np.errstate(divide='ignore', invalid='ignore'):
                transformed_columns[column + '_log'] = np.where(
                    df_reset[column] > 0, np.log10(df_reset[column]), np.nan)

            # Calculate the reciprocal
            transformed_columns[f"{column}_reciprocal"] = np.where(
                df_reset[column] != 0, 1 / df_reset[column], np.nan)

            # Calculate the square
            transformed_columns[f"{column}_squared"] = df_reset[column] ** 2

    # Concatenate the transformed columns into a single DataFrame
    transformed_df = pd.DataFrame(transformed_columns, index=df_reset.index)

    return transformed_df


def calc_continuous_feat_values(col_list, paml_feat, checklist, feat_set="paml"):
    '''Create the paml feature table for gene pairs in the `genes` file.'''

    cols = [col for col in paml_feat.loc[:, col_list].columns
            if not ('_binned' in col or '_log' in col
                    or '_reciprocal' in col or '_squared' in col)]  # columns to calculate feature values for

    # Get the feature values for each gene
    feature_values = paml_feat[cols].T.to_dict()  # genes as keys
    gene1_values = genes.gene1.map(feature_values)
    gene2_values = genes.gene2.map(feature_values)

    # Create feature value dataframes that correspond to the gene pairs in genes
    na_vals = {col: np.nan for col in cols}
    gene1_values = gene1_values.apply(lambda x: na_vals if pd.isna(x) else x)
    gene1_values_df = pd.DataFrame.from_records(
        gene1_values.values, index=genes.gene1)
    gene2_values = gene2_values.apply(lambda x: na_vals if pd.isna(x) else x)
    gene2_values_df = pd.DataFrame.from_records(
        gene2_values.values, index=genes.gene2)

    cols_features = []
    for i, row in tqdm(checklist.iterrows(), total=len(checklist), desc="Processing features"):
        if feat_set == "paml":
            if row["Feature name"] in ['Ka', 'Ka/Ks', 'Ks']:
                cols_values = calc_feature_values(gene1_values_df, gene2_values_df,
                                                  cols, row["Calculation for gene pair"])
                cols_features.append(cols_values)
                del cols_values
        else:
            if row["Feature name"] in ['Gene family size', 'Lethality binary',
                                       'Lethality score', 'Retention rate']:
                cols_values = calc_feature_values(gene1_values_df, gene2_values_df,
                                                  cols, row["Calculation for gene pair"])
                cols_features.append(cols_values)
                del cols_values

    print("Combine features into a single dataframe...")
    cols_features_df = pd.concat(cols_features, axis=1, ignore_index=False)

    # Remove duplicate columns (not sure why this happens)
    cols_features_df = cols_features_df.loc[:,
                                            ~cols_features_df.columns.duplicated()]

    # cols_features.index = pd.MultiIndex.from_tuples(list(cols_features.index), names=["gene1", "gene2"])
    del cols_features

    # Apply transformations
    cols_features_df = apply_transformations_df(cols_features_df)

    return cols_features_df


if __name__ == '__main__':
    os.chdir('/home/seguraab/ara-kinase-prediction/')

    print("Reading the data...")
    # Feature checklist file
    checklist = pd.read_csv(
        'data/Features/02_evolutionary_properties_feature_list.csv')

    # Open the single gene feature tables
    feat1 = pd.read_csv(
        'data/Features/evolutionary_properties_single_genes_feature_table_part1.csv', index_col=0)
    paml_feat = dt.fread(
        'data/Features/evolutionary_properties_single_genes_feature_table_part2_paml_only.csv').to_pandas()
    blast_hits = dt.fread(
        'data/Features/evolutionary_properties_single_genes_reciprocal_best_match_adjacency_matrix.csv.gz').to_pandas()
    blast_hits.set_index('gene', inplace=True)
    blast_hits = blast_hits.astype(int)  # convert boolean to binary

    print("Preparing the data...")
    # Calculate the median dN, dS, and dN/dS values across splice variants of a gene
    paml_feat["gene"] = paml_feat["C0"].str.split(
        '.').str[0]  # get gene ID from mRNA ID
    paml_feat = paml_feat.drop(columns="C0").groupby("gene").median()

    # If any of the splice variants for two genes were reciprocal blast hits, set their value to 1
    # row and col gene names are the same
    sum(blast_hits.columns == blast_hits.index)
    row_groups = blast_hits.index.str.split('.').str[0]
    col_groups = blast_hits.columns.str.split('.').str[0]
    blast_hits_grouped = (blast_hits.groupby(row_groups, axis=0).any().
                          groupby(col_groups, axis=1).any()).astype(int)
    blast_dict = blast_hits_grouped.to_dict()

    print("Reading in the gene pairs...")
    # Instances file
    # genes = pd.read_csv('data/instances_dataset_1.txt', sep='\t')
    # genes = pd.read_csv('data/2021_cusack_data/Dataset_4.txt', sep='\t')
    # genes = genes["pair_ID"].str.split("_", expand=True)
    # genes.columns = ['gene1', 'gene2']
    # genes = pd.read_csv('data/Kinase_genes/instances_tair10_kinases.txt', sep='\t')
    genes = pd.read_csv(
        'data/20250403_melissa_ara_data/corrected_data/binary_labels_from_linear_model.csv')

    ###################### Calculate the features values #######################
    features = {}

    print("Make the paml features...")
    ka_ks_cols = paml_feat.columns.str.contains('dN/dS')
    ks_cols = paml_feat.columns.str.contains('_dS_')
    ka_cols = paml_feat.columns.str.contains('_dN_')
    features['Ka/Ks'] = calc_continuous_feat_values(
        ka_ks_cols, paml_feat, checklist)
    features['Ks'] = calc_continuous_feat_values(ks_cols, paml_feat, checklist)
    features['Ka'] = calc_continuous_feat_values(ka_cols, paml_feat, checklist)

    print("Make the reciprocal best match features...")
    features['Reciprocal best match'] = calc_binary_adj_feat_values(
        blast_hits_grouped, "binary_reciprocal_best_match", genes)

    # Make the rest of the features
    feat_name = ['Gene family size', 'Lethality binary', 'Lethality score',
                 'Retention rate']
    col_list = ['continuous_gene_family_size', 'binary_lethality',
                'continuous_lethality_score', 'continuous_retention_rate']
    for i, col in enumerate(col_list):
        print(f"Make the {feat_name[i]} features...")
        features[feat_name[i]] = calc_continuous_feat_values(
            [col], feat1, checklist, feat_set='')

    # Clear the memory
    del paml_feat, blast_hits, blast_hits_grouped, blast_dict, feat1

    # Save the final feature tables
    # Note: pd.concat gets killed because of memory issues when having 2.7M rows.
    for key, table in features.items():
        print(key, table.shape)
        print(table.isna().sum(axis=0) / table.shape[0])  # % missing data
        # many gene pairs don't have any features
        if key != 'Reciprocal best match':
            table.index = pd.MultiIndex.from_arrays(
                [genes['gene1'].values, genes['gene2'].values], names=["gene1", "gene2"])
        # table.to_csv(
        # 	f'data/2021_cusack_data/Dataset_4_Features/Dataset_4_features_evolutionary_properties_{key.replace("/", "_")}.txt',
        # 	sep='\t', chunksize=100, index=True)
        # table.to_csv(
        #     f'data/Kinase_genes/features/TAIR10_kinases_features_evolutionary_properties_{key.replace("/", "_").replace(' ', '_')}.txt',
        #     sep='\t', chunksize=100, index=True)
        table.to_csv(
            f'data/20250403_melissa_ara_data/features/20250403_melissa_ara_features_for_binary_clf_evolutionary_properties_{key.replace("/", "_").replace(" ", "_")}.txt',
            sep='\t', index=True)
    # pd.concat(features.values(), axis=1, ignore_index=False).to_csv(
    # 	'data/Kinase_genes/features/TAIR10_kinases_features_evolutionary_properties_updated.txt',
    # 	sep='\t', chunksize=100)
    # ddf = dd.from_pandas(pd.concat(features.values(), axis=1, ignore_index=False), npartitions=10)
    # ddf.to_csv('data/Kinase_genes/features/TAIR10_kinases_features_evolutionary_properties_updated.txt', sep='\t')

    # Update the checklist, since I made more features than in Cusack et al. 2021 original checklist
    # Note: use the columns from TAIR10_kinases_features_evolutionary_properties_updated.txt to update the checklist

    # ml_features = checklist.groupby(
    # 	['NEW Feature name', 'Calculation for gene pair', 'Data processing method'])\
    # 	['ML model feature name'].apply(list).reset_index()

    # for idx, row in tqdm(ml_features.iterrows()):
    # 	ml_feat_names = row['ML model feature name']
    # 	feat_name = row['NEW Feature name']
    # 	calc_type = row['Calculation for gene pair']
    # 	feat_type = row['Data processing method']
    # 	print(f'Generating features for {feat_name}')

    # 	if feat_name in ['0 blast res', 'tbd', 'arabidopsis paml combined results']:
    # 		continue

    # 	elif feat_name in ['gene family size', 'retention rate']:
    # 		target_col = '_'.join([feat_type.lower(), feat_name.replace(' ', '_')])
    # 		values = calc_feature_values(gene1_values, gene2_values,
    # 			genes, pd.DataFrame(feat1[target_col]), calc_type)

    # 	elif (feat_name == 'lethality') & (feat_type == 'Continuous'):
    # 		values = calc_feature_values(gene1_values, gene2_values,
    # 			genes, pd.DataFrame(feat1['continuous_lethality_score']), calc_type)

    # 	elif (feat_name == 'lethality') & (feat_type == 'Binary'):
    # 		values = calc_feature_values(gene1_values, gene2_values,
    # 			genes, pd.DataFrame(feat1['binary_lethality']), calc_type)

    # 	values = pd.DataFrame.from_dict(values, orient="index") #type: ignore
    # 	values = pd.concat([values] * len(ml_feat_names), axis=1)
    # 	values.columns = ml_feat_names
    # 	values.index = pd.MultiIndex.from_tuples(list(values.index), names=["gene1", "gene2"]) #type: ignore
    # 	values = apply_transformations_df(values)
    # 	features[f"{feat_name}_{calc_type}"] = values
    # 	del values

    # Save the feature table to a file!
    # pd.concat(features.values(), axis=1, ignore_index=False).to_csv(
    # 	'data/Features/evolutionary_properties_gene_pairs_features.txt', sep='\t')
    # pd.concat(features.values(), axis=1, ignore_index=False).to_csv(
    # 	'data/2021_cusack_data/Dataset_4_Features/Dataset_4_features_evolutionary_properties.txt',
    # 	sep='\t')
    # pd.concat(features.values(), axis=1, ignore_index=False).to_csv(
    # 	'data/Kinase_genes/features/TAIR10_kinases_features_evolutionary_properties.txt',
    # 	sep='\t', chunksize=100)

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
