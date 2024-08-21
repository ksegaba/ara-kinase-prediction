#!/usr/bin/env python3
'''Create the network properties feature table for single genes as instances.'''
# %%
import json
import pandas as pd
import numpy as np
import networkx as nx

__author__ = 'Kenia Segura Abá'

# Feature checklist file
checklist = pd.read_csv('../data/2021_cusack_data/06_network_properties_feature_list.csv')

# Open the JSON files in the checklist
aranet = json.load(open(checklist['File path'].unique()[0])) # AraNet genetic interactions
co_expr = json.load(open(checklist['File path'].unique()[1])) # Co-expression network
ppi = json.load(open(checklist['File path'].unique()[2])) # Protein-protein interactions

# Instances file
genes = pd.read_csv('/home/seguraab/ara-kinase-prediction/data/instances_dataset_singles.txt')

# %%
########################## Create the AraNet features ##########################
# The number of interactions as a feature
genes.insert(1, 'NumInteractions', genes.apply(lambda x: len(aranet[x['gene']]) \
             if x['gene'] in aranet.keys() else np.nan, axis=1))

# Adjacency matrix features
G = nx.Graph()
for key, values in aranet.items():
    for value in values:
        G.add_edge(key, value)

adj_mat = nx.to_pandas_adjacency(G)
adj_mat.index.name = 'gene'
# adj_mat.to_csv('../data/Features/network_properties_single_genes_aranet_adjacency_matrix.csv.gz', compression='gzip')

# check to make sure the number of edges add up to the number of interactions (NumInteractions)
adj_mat.sum()
# adj_mat.iloc[0,:].loc[adj_mat.iloc[0,:]==1].index
# aranet['AT1G06190']
'''Note: The reason why adj_mat.sum() does not match NumInteractions in genes, 
is because the dictionary values are missing some of the interacting gene pairs.
For example, aranet['AT1G06190'] is missing "AT1G04870" and "AT1G02150" but in
the values, but aranet['AT1G04870'] and aranet['AT1G02150'] have "AT1G06190" in 
their values. Thus, I will use networkx to calculate the number of interactions 
for each node (i.e. gene).'''

# Calculate the number of interactions for each node (gene)
num_int = pd.DataFrame.from_dict(dict(G.degree()), orient='index')
num_int.columns = ['NumInteractions']
num_int.index.name = 'gene'
# num_int.to_csv('../data/Features/network_properties_single_genes_aranet_num_interactions.csv')
genes.drop(columns='NumInteractions', inplace=True)

del G, adj_mat, num_int

# %%
################### Create the co-expression network features ##################
# Convert the JSON into a dataframe and sort the index

# The feature values are the assigned co-expression cluster numbers for each gene
co_expr_feat = pd.DataFrame.from_dict(co_expr, orient='index').sort_index().T
co_expr_feat.index.name = 'gene'
co_expr_feat.to_csv('../data/Features/network_properties_single_genes_co_expression_clusters.csv')

# One-hot encode the co-expression cluster numbers
encoded_co_expr = {}
for col in co_expr_feat.columns:
    encoded_co_expr[col] = pd.get_dummies(co_expr_feat[col], prefix=f'{col}_cluster')

pd.concat(encoded_co_expr.values(), ignore_index=False, axis=1).astype('int').to_csv(
    '../data/Features/network_properties_single_genes_co_expression_clusters_onehot.csv.gz',
    compression='gzip', chunksize=1000)

del co_expr_feat, encoded_co_expr

# %%
################ Create the protein-protein interaction features ###############
# Adjacency matrix features
G = nx.Graph()
for key, values in ppi.items():
    for value in values:
        G.add_edge(key, value)

adj_mat = nx.to_pandas_adjacency(G)
adj_mat.index.name = 'gene'
# adj_mat.to_csv('../data/Features/network_properties_single_genes_ppi_adjacency_matrix.csv.gz', compression='gzip')

# check to make sure the number of edges add up to the number of interactions (NumInteractions)
genes.insert(1, 'NumInteractions', genes.apply(lambda x: len(ppi[x['gene']]) \
             if x['gene'] in ppi.keys() else np.nan, axis=1))

adj_mat.sum()
# adj_mat.iloc[0,:].loc[adj_mat.iloc[0,:]==1].index
# ppi['AT4G02110']
# ppi['AT4G16760']
'''Note: The same thing as with AraNet is occurring with the PPI data.'''

# Calculate the number of interactions for each node (gene)
num_int = pd.DataFrame.from_dict(dict(G.degree()), orient='index')
num_int.columns = ['NumInteractions']
num_int.index.name = 'gene'
# num_int.to_csv('../data/Features/network_properties_single_genes_ppi_num_interactions.csv')
genes.drop(columns='NumInteractions', inplace=True)
