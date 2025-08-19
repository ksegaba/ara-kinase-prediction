#!/usr/bin/env python
'''Create the evolutionary properties feature table for single genes as instances.'''
# %%
import os, json
import numpy as np
import pandas as pd
import datatable as dt
import networkx as nx
from tqdm import tqdm

__author__ = 'Kenia Segura Abá'

# %% Functions for manipulating dictionaries
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


def subset_dict_vals(dictionary, val_idx, remove_keys=[]):
	'''Subset the values of a dictionary where the value is a list.
	Parameters:
		- dictionary (dict): A dictionary with keys and values (list)
		- val_idx (int): The index of the element of interest in the value list
		- remove_keys (list): A list of keys to exclude from the subset
	'''
	subset_dict = {}
	for key, value in dictionary.items():
		if key not in remove_keys:
			if isinstance(value, list):
				subset_dict[key] = value[val_idx]

	return subset_dict


# %%
def make_blast_adj():
	'''Create an adjacency matrix of Arabidopsis thaliana gene pairs with their 
	top 2 hits in a BLASTp search against itself. `blast_hits` is a set of gene 
	pairs that include self-hits and the next best hit in the BLASTp search.'''
	
	# Load the blast results
	path = 'data/evolutionary_properties_data/0_blast_res'
	blast_res = pd.read_csv(f'{path}/TAIR10_self_blastp.txt', sep='\t', header=None)
	blast_res[1] = blast_res[1].apply(lambda x: x.split(' ')[0]) # get subject gene ID
	
	# Determine which genes have consistent matches
	blast_hits = {tuple(sorted([gene1, gene2])) for gene1, gene2 in \
			   zip(blast_res[0], blast_res[1])} # unique gene pairs only
	
	# Create the adjacency matrix
	G = nx.Graph()
	G.add_edges_from(blast_hits)
	adj = pd.DataFrame(nx.adjacency_matrix(G).todense(), index=G.nodes, columns=G.nodes)
	# adj.to_csv('data/evolutionary_properties_data/0_blast_res/TAIR10_self_best_matches_adj_mat.csv',
	# 		chunksize=100)
	del adj

	# Map RefSeq gene IDs to TAIR10 IDs
	adj = dt.fread('data/evolutionary_properties_data/0_blast_res/TAIR10_self_best_matches_adj_mat.csv').to_pandas()
	id_map = pd.read_csv('data/NCBI_genomes/GCF_000001735.4/TAIR10_NCBI_REFSEQ_mapping_PROT',
				sep='\t', names=['Num_ID', 'NCBI_ID', 'TAIR10_ID']) # NCBI identifier mapping
	id_map = id_map[['NCBI_ID', 'TAIR10_ID']].set_index('NCBI_ID').to_dict() # dictionary of NCBI to TAIR10 IDs
	adj = adj.rename(columns=id_map['TAIR10_ID']) # rename columns to TAIR10 IDs
	adj['C0'] = adj['C0'].apply(lambda x: id_map['TAIR10_ID'][x] if x in id_map['TAIR10_ID'].keys() else x)
	adj.rename(columns={'C0': 'gene'}, inplace=True)
	
	# Save the feature table
	# adj.to_csv(f'data/Features/evolutionary_properties_single_genes_reciprocal_best_match_adjacency_matrix.csv.gz',
	# 		 index=False, compression='gzip', chunksize=100)
	
	# Note: I did not add the adjacency matrix to the feature table `df_list`
	
	return None


# %% Functions to generate PAML feature values
def get_cds_genes(gff):
	'''Get the list of CDS gene identifiers from a GFF file.'''
	
	df = dt.fread(gff, fill=True, skip_to_line=9)
	df = df[dt.f.C2 == 'CDS', 'C8'] # CDS gene descriptions
	df = df.to_pandas()
	genes = df.C8.str.split(';', expand=True)[0].\
			str.replace('ID=cds-', '').unique().tolist()
	
	return genes


def make_paml_df():
	paml_df = pd.json_normalize(paml, sep='//')
	paml_df.columns = pd.MultiIndex.from_tuples([tuple(col.split('//')) for col in paml_df.columns])
	
	# Assign the species to each arabidopsis gene match used to calculate dN, dS, and dN/dS
	paml_list = []
	for ara_gene in tqdm(paml_df.columns.get_level_values(0).unique()):
		# convert the dictionary entry into a dataframe
		gene_df = paml_df[ara_gene].stack([0,1], future_stack=True).reset_index()
		
		# assign the species to each gene match
		gene_df['level_3'] = gene_df.apply(lambda x: 'Sly' if x['level_2'] in sly_genes \
			else 'Ptr' if x['level_2'] in ptr_genes \
			else 'Gma' if x['level_2'] in gma_genes \
			else 'Tca' if x['level_2'] in tca_genes else 'Ath', axis=1)
		gene_df['new_cols'] = gene_df['level_1'] + '_' + gene_df['level_3']
		gene_df = gene_df[['new_cols', 'NG86', 'YN00', 'LWL85', 'LWL85m', 'LPB93']].T
		gene_df.index = pd.MultiIndex.from_tuples([(ara_gene, row) for row in gene_df.index])
		
		paml_list.append(gene_df)
		del gene_df

	# combine the values of new_cols into the column names, so NG86 becomes NG86_Ath, NG86_Ptr, etc.
	paml_final_df = pd.concat(paml_list, axis=0)
	paml_final_df.columns = paml_final_df.iloc[0].values
	paml_final_df = paml_final_df.iloc[1:]
	# paml_final_df.to_csv('data/evolutionary_properties_data/3_paml_res/arabidopsis_paml_combined_results_values.csv', chunksize=1000)

	return paml_final_df


# %% General functions to generate feature values for evolutionary properties
def apply_transformations(values, method):
	''' Apply transformations to a dataframe of continuous feature values.
	
	Parameters:
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
		print('No transformation applied')
		return values
	
	if method == 'Continuous':
		df_list = []
		# Bin the feature value
		binned_values = pd.qcut(values, q=4, labels=False, duplicates='drop')
		binned_values.name = values.name + '_binned'
		
		# Calculate the log
		log_values = values.apply(lambda x: np.log10(x) if x > 0 else np.nan)
		log_values.name = values.name + '_log'
		
		# Calculate the reciprocal
		rec_values = values.apply(lambda x: 1/x if x != 0 else np.nan)
		rec_values.name = values.name + '_reciprocal'
		
		# Calculate the square
		sqr_values = values ** 2
		sqr_values.name = values.name + '_squared'
		
		# Combine the transformed values into a single dataframe
		df_list.extend([values, binned_values, log_values, rec_values, sqr_values])
		new_values = pd.concat(df_list, axis=1)
		
		return new_values


def calc_feature_values(json, calc_type, col_name):
	if calc_type == 'Binary':
		values = pd.DataFrame.from_dict(json, orient='index')
		values.columns = [col_name]
		return values
	
	if calc_type == 'Continuous':
		values = pd.to_numeric(pd.Series(json, name=col_name))
		new_values = apply_transformations(values, method='Continuous')
		return new_values


# %%
if __name__ == '__main__':
	os.chdir('/home/seguraab/ara-kinase-prediction/')

	# Feature checklist file
	checklist = pd.read_csv('data/Features/02_evolutionary_properties_feature_list.csv')
	checklist['File path'].unique()

	# Instances file
	genes = pd.read_csv('data/instances_dataset_singles.txt')

	# Gene lists for each species
	ath_genes = get_cds_genes('data/NCBI_genomes/GCF_000001735.4/genomic.gff')
	ptr_genes = get_cds_genes('data/NCBI_genomes/GCF_000002775.5/genomic.gff')
	gma_genes = get_cds_genes('data/NCBI_genomes/GCF_000004515.6/genomic.gff')
	sly_genes = get_cds_genes('data/NCBI_genomes/GCF_000188115.5/genomic.gff')
	tca_genes = get_cds_genes('data/NCBI_genomes/GCF_000208745.1/genomic.gff')

	# Open the BLAST and JSON files in the checklist
	gfam = json.load(open(
		'data/2021_cusack_data/21_arabidopsis_redundancy/02_evolutionary_properties/gene_family_size/gene_family_size_dictionary.json'))
	paml = json.load(open(
		'data/evolutionary_properties_data/3_paml_res/arabidopsis_paml_combined_results.json'))
	pheno = json.load(open(
		'data/2021_cusack_data/21_arabidopsis_redundancy/02_evolutionary_properties/lethality_score/lethality_dict_121817.json'))
	reten = json.load(open(
		'data/2021_cusack_data/21_arabidopsis_redundancy/02_evolutionary_properties/retention_rate/Retention_rate_dictionary.json'))
	
	make_blast_adj() # Create the adjacency matrix of consistent BLAST hits
	# blast_hits = dt.fread(
	# 	'data/Features/evolutionary_properties_single_genes_reciprocal_best_match_adjacency_matrix.csv.gz').\
	# 	to_pandas() # Load the adjacency matrix
	
	make_paml_df() # Get feature values from paml
	paml_df = dt.fread(
		'data/evolutionary_properties_data/3_paml_res/arabidopsis_paml_combined_results_values.csv').\
		to_pandas() # Load the feature values from paml
	
	# Create the feature table
	feat_list = []
	paml_feat_list = []
	for idx, row in checklist[['File path', 'NEW Feature name', 'Data processing method']].drop_duplicates().iterrows():
		if row['NEW Feature name'] == 'gene family size':
			feat_list.extend([calc_feature_values(gfam, 'Continuous', 'continuous_gene_family_size')])
		
		elif row['NEW Feature name'] == 'arabidopsis paml combined results':
			for feat_set in ['NG86', 'YN00', 'LWL85', 'LWL85m', 'LPB93']:
				sub_df = paml_df[paml_df['C1'] == feat_set]
				sub_df.drop('C1', axis=1, inplace=True)
				sub_df.set_index('C0', inplace=True)
				sub_df.index.name = 'gene'
				sub_df.columns = 'continuous_' + sub_df.columns + '_' + feat_set # set new column names
				
				for col in sub_df.columns: # apply transformations
					try:
						paml_feat_list.append(apply_transformations(sub_df[col].astype('float'), 'Continuous'))
					except ValueError:
						sub_df[col].replace('', np.nan, inplace=True)
						paml_feat_list.append(apply_transformations(sub_df[col].astype('float'), 'Continuous'))
					
				del sub_df
			
		elif row['NEW Feature name'] == 'lethality':
			if row['Data processing method'] == 'Binary':
				pheno_bin = subset_dict_vals(pheno, val_idx=1, remove_keys=['Gene'])
				pheno_bin = {key: 1 if val == 'Yes' else 0 for key, val in pheno_bin.items()} # convert to binary
				feat_list.extend([calc_feature_values(pheno_bin, 'Binary', 'binary_lethality')])
			
			elif row['Data processing method'] == 'Continuous':
				pheno_cont = subset_dict_vals(pheno, val_idx=0, remove_keys=['Gene'])
				feat_list.extend([calc_feature_values(pheno_cont, 'Continuous', 'continuous_lethality_score')])
		
		elif row['NEW Feature name'] == 'retention rate':
			feat_list.extend([calc_feature_values(reten, 'Continuous', 'continuous_retention_rate')])
	
	# Save feature tables
	pd.concat(feat_list, axis=1, ignore_index=False).to_csv(
		'data/Features/evolutionary_properties_single_genes_feature_table_part1.csv')
	paml_feat_df = pd.concat(paml_feat_list, axis=1, ignore_index=False)

	id_map = pd.read_csv('data/NCBI_genomes/GCF_000001735.4/TAIR10_NCBI_REFSEQ_mapping_PROT',
				sep='\t', names=['Num_ID', 'NCBI_ID', 'TAIR10_ID']) # NCBI identifier mapping
	id_map = id_map[['NCBI_ID', 'TAIR10_ID']].set_index('NCBI_ID').to_dict() # dictionary of NCBI to TAIR10 IDs
	paml_feat_df.index = paml_feat_df.apply(lambda x: id_map['TAIR10_ID'][x.name] \
		if x.name in id_map['TAIR10_ID'].keys() else x.name, axis=1).values
	
	paml_feat_df.to_csv('data/Features/evolutionary_properties_single_genes_feature_table_part2_paml_only.csv')
	
	# To Do: Incorporate the whole genome duplication features and update the checklist file
# %%
	############################################################################
	'''Confirm that wgdDupGenes.MLD_dictionary.json contains everything in 
	betaGammaWGD.MLD_dictionary.json plus extra gene pairs.'''
	general_wgd = json.load(open('data/2021_cusack_data/21_arabidopsis_redundancy/02_evolutionary_properties/duplication_event/wgdDupGenes.MLD_dictionary.json', 'r'))
	beta_wgd = json.load(open('data/2021_cusack_data/21_arabidopsis_redundancy/02_evolutionary_properties/duplication_event/betaGammaWGD.MLD_dictionary.json', 'r'))

	gen_wgd_df = pd.DataFrame.from_dict(general_wgd, orient='index') # 27440 rows, all unique genes
	bet_wgd_df = pd.DataFrame.from_dict(beta_wgd, orient='index') # 27440 rows, all unique genes

	bet_wgd_df.apply(lambda x: x.name in gen_wgd_df.index, axis=1).sum() # returns 27440
# %%
	############################################################################
	# Check which genes have lethality scores
	dict_val_counts(pheno, val_idx=1) # returns {'No': 24486, 'Yes': 2720, 'Predicted lethal?': 1}
	
	'''Note on pheno: I believe the values are in the list are repeated, so there
	are actually 2, not 4 unique values'''
	# lengths=[]
	# for key, valu in pheno.items():
	# 	lengths.append(len(set(valu)))
	# set(lengths) # returns 2, so there are only 2 unique values for each key
	############################################################################