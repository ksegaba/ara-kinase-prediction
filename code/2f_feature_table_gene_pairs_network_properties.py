#!/usr/bin/env python3
'''Create the network properties feature table for gene pairs as instances.'''
# %%
import json
import os
import pandas as pd
import datatable as dt
import numpy as np
import networkx as nx

__author__ = 'Kenia Segura Abá'

os.chdir('/home/seguraab/ara-kinase-prediction')


def calculate_feature_values(df, gene_pairs, feature_type):
    ''' Generate a feature table for a given set of instances (gene pairs) from 
    the data in df, depending on the feature type (feature_type). Feature values 
    are based on the 'Calculation for gene pair' column in the checklist file

    If the feature type is 'continuous' then feature values are calculated from 
    an m x m adjacency matrix (df), where m is the  number of instances. Instance 
    identifiers need to be set as the index and the column names of the dataframe.

    If the feature type is 'binary' then feature values are calculated from an 
    m x n matrix (df) of categorical values. The instance identifiers should be 
    set as the index of the dataframe and the columns are categorical variables.

    Parameters:
        - df (pd.DataFrame): data matrix to calculate feature values froms
        - gene_pairs (pd.DataFrame): instance identifiers in two columns
        - feature_type (str): type of feature to calculate feature values 
                              for ('continuous' or 'binary')

    Returns a dataframe with the following columns:
        If the feature_type is 'continuous':
            - gene1 (str): gene 1 identifiers
            - gene2 (str): gene 2 identifiers
            - overlap (float): number of overlapping interactors between gene1 and gene2
            - total (float): total number of unique interactors for gene1 and gene2
            - percent overlap (float): percentage of overlapping interactors

        If the feature_type is 'binary':
            - gene1 (str): gene 1 identifiers
            - gene2 (str): gene 2 identifiers
            - additional columns: binary values indicating whether gene1 and gene2
                                  have the same value in the columns of df. The 
                                  column names are the same as the column names 
                                  in df.
    '''

    # Create feature values for continuous feature types
    if feature_type == 'continuous':
        G = nx.from_pandas_adjacency(df)  # make graph

        features = []
        for i in range(len(gene_pairs)):
            gene1 = gene_pairs.gene1[i]
            gene2 = gene_pairs.gene2[i]

            # Calculate feature values
            try:
                overlap = len(set(nx.common_neighbors(G, gene1, gene2)))
                total = len(set(list(G.neighbors(gene1)) +
                            list(G.neighbors(gene2))))
                percent = (overlap/total) * 100
                features.append([gene1, gene2, overlap, total, percent])

            except nx.exception.NetworkXError:
                # At least one gene identifier is not in the network
                features.append([gene1, gene2, np.nan, np.nan, np.nan])

        col_names = ['gene1', 'gene2', '# overlapping',
                     'Total #', '% overlapping']

        return pd.DataFrame(features, columns=col_names)

    # Create feature values for binary feature types (same or not)
    elif feature_type == 'binary':
        features = []
        for i in range(len(gene_pairs)):
            gene1 = gene_pairs.gene1[i]
            gene2 = gene_pairs.gene2[i]

            # Determine if the two genes have the same annotation/property
            try:
                features.append(df.loc[gene1].eq(df.loc[gene2]).astype('int'))

            except KeyError:
                # At least one gene identifier is not in the network
                features.append(
                    pd.Series(np.nan, index=np.arange(len(df.columns))))

        return pd.concat([gene_pairs, pd.concat(features, axis=1).T], axis=1)

    else:
        raise ValueError(
            'Feature type must be either "continuous" or "binary".')


def apply_transformations(df):
    ''' Apply transformations to a dataframe of continuous feature values.

    Parameters:
        - df (pd.DataFrame): A data frame of feature values

    Returns an expanded dataframe with additional transformations on continuous columns:
        - Binned values
        - Logarithm
        - Reciprocal
        - Squared
    '''
    df_list = []

    # Apply the transformations to individual columns
    for column in df.columns:
        if column in ['# overlapping', '% overlapping', 'Total #']:
            # Bin the feature value
            bins = pd.qcut(df[column], q=4, labels=False, duplicates='drop')
            bins.name = column + '_bin'

            # Calculate the log
            log_vals = df[column].apply(
                lambda x: np.log10(x) if x > 0 else np.nan)
            log_vals.name = column + '_log'

            # Calculate the reciprocal
            reciprocal = df[column].apply(lambda x: 1/x if x != 0 else np.nan)
            reciprocal.name = column + '_reciprocal'

            # Calculate the square
            square = df[column] ** 2
            square.name = column + '_squared'

            df_list.append(pd.concat([df[column], bins, log_vals, reciprocal,
                                      square], axis=1))

        else:
            # No transformations will be applied to binary features
            df_list.append(df[column])

    return pd.concat(df_list, axis=1)


def assign_feature_column_names(df, feature_name):
    ''' Assign feature column names to the columns of a dataframe according to 
    the feature engineering checklist file.

    Parameters:
        - df (pd.DataFrame): dataframe of feature values
        - feature_name (str): name of the feature category in the 'Feature name'
                              column of the checklist file

    Returns: 
        - col_names (dict): a dictionary containing the original column names 
                            as keys and the new column names as values.
        - col_to_drop (list): a list of column names to drop from the dataframe
    '''
    col_names = {}  # column name mapping
    col_to_drop = []  # columns to drop
    subset = checklist[checklist['NEW Feature name'] == feature_name]

    if feature_name in ['aranet', 'ppi']:
        # Determine the new column names for df
        for col in df.columns:
            if col.split('_')[0] in subset['Calculation for gene pair'].unique():
                if col.endswith('_bin'):
                    new_col = subset.loc[(subset['NEW Feature name'] == feature_name) &
                                         (subset['Calculation for gene pair']
                                          == col.split('_')[0])
                                         & (subset['Transformation'] == 'Binned'),
                                         'ML model feature name'].values[0]

                elif col.endswith('_log'):
                    new_col = subset.loc[(subset['NEW Feature name'] == feature_name) &
                                         (subset['Calculation for gene pair']
                                          == col.split('_')[0])
                                         & (subset['Transformation'] == 'Log'),
                                         'ML model feature name'].values[0]

                elif col.endswith('_reciprocal'):
                    new_col = subset.loc[(subset['NEW Feature name'] == feature_name) &
                                         (subset['Calculation for gene pair']
                                          == col.split('_')[0])
                                         & (subset['Transformation'] == 'Reciprocal'),
                                         'ML model feature name'].values[0]

                elif col.endswith('_squared'):
                    new_col = subset.loc[(subset['NEW Feature name'] == feature_name) &
                                         (subset['Calculation for gene pair']
                                          == col.split('_')[0])
                                         & (subset['Transformation'] == 'Squared'),
                                         'ML model feature name'].values[0]

                else:
                    new_col = subset.loc[(subset['NEW Feature name'] == feature_name) &
                                         (subset['Calculation for gene pair'] == col)
                                         & (subset['Transformation'].isna()),
                                         'ML model feature name'].values[0]

                col_names[col] = new_col

            else:
                col_names[col] = col

    if feature_name == 'all clust':
        for col in df.columns:
            try:
                new_col = subset['ML model feature name'].\
                    loc[subset['ML model feature name'].str.contains(
                        col)].values[0]
                col_names[col] = new_col

            except IndexError:
                if col in ['gene1', 'gene2']:
                    col_names[col] = col
                else:
                    col_to_drop.append(col)

    return col_names, col_to_drop


# %%
# Feature checklist file
checklist = pd.read_csv('data/Features/06_network_properties_feature_list.csv')

# Instances file
# gene_pairs = pd.read_csv('/home/seguraab/ara-kinase-prediction/data/instances_dataset_1.txt', sep='\t')
# gene_pairs = pd.read_csv('data/2021_cusack_data/Dataset_4.txt', delimiter='\t', header=0)
gene_pairs = pd.read_csv(
    'data/Kinase_genes/instances_tair10_kinases.txt', sep='\t')
# gene_pairs = gene_pairs["pair_ID"].str.split("_", expand=True)
# gene_pairs.columns = ['gene1', 'gene2']

# %%
# Create the network properties feature table
for feature_name in checklist['NEW Feature name'].unique():
    if feature_name == 'aranet':
        print("Creating the AraNet features...")
        adj_mat = dt.fread(
            'data/Features/network_properties_single_genes_aranet_adjacency_matrix.csv.gz').to_pandas()
        adj_mat.set_index('gene', inplace=True)

        aranet_features = calculate_feature_values(
            adj_mat, gene_pairs, 'continuous')  # calculate continuous feature values
        aranet_features = apply_transformations(
            aranet_features)  # apply transformations
        col_map, col_to_drop = assign_feature_column_names(
            aranet_features, 'aranet')  # get column names from the checklist file
        # rename the columns according to mapping
        aranet_features.rename(columns=col_map, inplace=True)

        if len(col_to_drop) > 0:  # drop columns that are not in the checklist file
            aranet_features.drop(columns=col_to_drop, inplace=True)

        del adj_mat, col_map, col_to_drop
    #
    if feature_name == 'all clust':
        print("Creating the co-expression features...")
        co_expr = dt.fread(
            'data/Features/network_properties_single_genes_co_expression_clusters.csv').to_pandas()
        co_expr.set_index('gene', inplace=True)

        co_expr_features = calculate_feature_values(
            co_expr, gene_pairs, 'binary')  # calculate binary feature values
        col_map, col_to_drop = assign_feature_column_names(
            co_expr_features, 'all clust')  # get column names from the checklist file
        # rename the columns according to mapping
        co_expr_features.rename(columns=col_map, inplace=True)

        if len(col_to_drop) > 0:  # drop columns that are not in the checklist file
            co_expr_features.drop(columns=col_to_drop, inplace=True)

        del co_expr, col_map, col_to_drop
    #
    if feature_name == 'ppi':
        print("Creating the PPI features...")
        adj_mat = dt.fread(
            'data/Features/network_properties_single_genes_ppi_adjacency_matrix.csv.gz').to_pandas()
        adj_mat.set_index('gene', inplace=True)

        # calculate continuous feature values
        ppi_features = calculate_feature_values(
            adj_mat, gene_pairs, 'continuous')
        ppi_features = apply_transformations(
            ppi_features)  # apply transformations
        col_map, col_to_drop = assign_feature_column_names(
            ppi_features, 'ppi')  # get column names from the checklist file
        # rename the columns according to mapping
        ppi_features.rename(columns=col_map, inplace=True)

        if len(col_to_drop) > 0:  # drop columns that are not in the checklist file
            ppi_features.drop(columns=col_to_drop, inplace=True)

        del adj_mat, col_map

# %%
# Ensure the data frames have the gene pairs in the same order
if (aranet_features.gene1.equals(co_expr_features.gene1)) &\
    (aranet_features.gene1.equals(ppi_features.gene1)) &\
    (aranet_features.gene2.equals(co_expr_features.gene2)) &\
        (aranet_features.gene2.equals(ppi_features.gene2)):

    feat_table = pd.concat([aranet_features, co_expr_features.iloc[:, 2:],
                            ppi_features.iloc[:, 2:]], axis=1)

    print("Saving the feature table to data/Features/network_properties_gene_pairs_features.csv")
    # feat_table.to_csv('data/Features/network_properties_gene_pairs_features.csv', index=False)
    # feat_table.to_csv('data/2021_cusack_data/Dataset_4_Features/Dataset_4_features_network_properties.txt', sep='\t', index=False)
    feat_table.to_csv(
        'data/Kinase_genes/features/TAIR10_kinases_features_network_properties.txt', sep='\t', index=False)

    print("Feature table dimensions:", feat_table.shape)
    print("Checklist file dimensions:", checklist.shape)

else:
    print("Gene pairs are not in the same order. Please check the dataframes.")
