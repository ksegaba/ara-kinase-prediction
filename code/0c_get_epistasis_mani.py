# Method to calculate epistasis values using genotype estimated marginal means #

# %%
import itertools, os
import datatable as dt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


os.chdir('/home/seguraab/ara-kinase-prediction/code')


def plot_single_mutants(data, labels):
	# Plot the distributions of MA and MB for each trait
	for label in labels:
		MA_vals = data.loc[data.Genotype == 'MA', label]
		MB_vals = data.loc[data.Genotype == 'MB', label]
		fig = plt.figure()
		sns.histplot(MA_vals, color='blue', label='MA')
		sns.histplot(MB_vals, color='red', label='MB')
		plt.title(label)
		plt.legend()
		plt.show()
	
	return fig


def reshape_data(data, labels):
	'''Reshape estimated marginal means data to wide format for each label.'''
	
	data = data.loc[:, ['Set', 'Genotype'] + labels]
	data.drop_duplicates(inplace=True)

	## check each set has 4 genotypes
	tmp = data.groupby('Set')['Genotype'].value_counts().reset_index('Genotype')
	print(sum(tmp.groupby('Set')['Genotype'].nunique() == 4) == tmp.index.nunique()) # True
	del tmp

	dataw = data.pivot_table(index='Set', columns='Genotype', values=labels)
	dataw.shape # (127, 44)
	
	return dataw


def sort_single_mutants(dataw, labels):
	'''Sorts the values of MA and MB so that MA is always greater than MB.
	dataw: DataFrame in wide format with genotypes as columns and sets as rows.
	trait: The trait to sort values for (column multi-index level 0).'''
	
	sorted_df = dataw.copy(deep=True)
	for trait in labels:
		for row in sorted_df.index:
			if sorted_df.loc[row, (trait, 'MA')] < sorted_df.loc[row, (trait, 'MB')]:
				tmp_MA = sorted_df.loc[row, (trait, 'MA')]
				sorted_df.loc[row, (trait, 'MA')] = sorted_df.loc[row, (trait, 'MB')]
				sorted_df.loc[row, (trait, 'MB')] = tmp_MA

	return sorted_df


def calc_relative_fitness(sorted_df):
	'''Calculates relative fitness for each genotype based on the wild type (WT) genotype.'''

	W_df = sorted_df.copy(deep=True)
	# note, some sets have WT = 0, so the relative fitness will be undefined.
	for trait in W_df.columns.levels[0]:
		W_df.loc[:,trait] = W_df[trait].apply(lambda x: x / x['WT'], axis=1).values

	# Add W_ to column names to denote relative fitness
	W_df.columns = pd.MultiIndex.from_arrays(
		[W_df.columns.get_level_values(0).values,
		'W_' + W_df.columns.get_level_values(1).values])

	return W_df


def calc_epistasis(W_df, save_path):
	'''Calculates epistasis values based on 4 established genetic interaction
	definitions (Mani 2008) and additional definitions using the relative
	fitness values of the single mutant genotypes for each trait.'''

	for trait in W_df.columns.levels[0]:
		W_df[(trait, 'epi_min')] = W_df[trait].apply(
			lambda x: x['W_DM'] - min(x['W_MA'], x['W_MB']), axis=1).values
		W_df[(trait, 'epi_product')] = W_df[trait].apply(
			lambda x: x['W_DM'] - (x['W_MA'] * x['W_MB']), axis=1).values
		W_df[(trait, 'epi_additive')] = W_df[trait].apply(
			lambda x: x['W_DM'] - (x['W_MA'] + x['W_MB'] - 1), axis=1).values
		W_df[(trait, 'epi_log2_mani')] = W_df[trait].apply(
			lambda x: x['W_DM'] - np.log2((2**x['W_MA'] - 1) * (2**x['W_MB'] - 1) + 1), axis=1).values
		W_df[(trait, 'epi_mean')] = W_df[trait].apply(
			lambda x: x['W_DM'] - np.mean([x['W_MA'], x['W_MB']]), axis=1).values
		W_df[(trait, 'epi_max')] = W_df[trait].apply(
			lambda x: x['W_DM'] - max(x['W_MA'], x['W_MB']), axis=1).values
		W_df[(trait, 'epi_log2_additive')] = W_df[trait].apply(
			lambda x: x['W_DM'] - np.log2(x['W_MA'] * x['W_MB']), axis=1).values
		W_df[(trait, 'epi_log2_difference')] = W_df[trait].apply(
			lambda x: x['W_DM'] - np.log2(x['W_MA'] / x['W_MB']), axis=1).values

	W_df.to_csv(save_path, sep='\t')
	return W_df

# %%
'''Calculate epistasis values based on 4 established genetic interaction
definitions (Mani 2008) using the estimated marginal means of each label:
epistasis = W_ab - Expected(W_ab)
- Minimum: epistasis = W_ab - min(W_a, W_b)
- Product: epistasis = W_ab - W_a * W_b
- Additive: epistasis = W_ab - (W_a + W_b - 1)
- Log-additive: epistasis = W_ab - np.log2((2**W_a - 1) * (2**W_b - 1) + 1)'''

dir_path = '../data/20240923_melissa_ara_data/corrected_data'

for trait in ['PG', 'DTB', 'LN', 'DTF', 'SN', 'WO', 'FN', 'SPF', 'TSC', 'SH']:
	df = dt.fread(f'{dir_path}/fitness_data_for_Kenia_09232024_{trait}_emmeans.tsv')
	df = df.to_pandas()
	
	if trait in ['SN', 'WO', 'FN', 'TSC']:
		labels = [f'{trait}_emmean', f'{trait}_plus1_emmean',
				  f'{trait}_plus1_log10_emmean', f'{trait}_plog10_emmean']
	else:
		labels = [f'{trait}_emmean', f'{trait}_log10_emmean',
				  f'{trait}_plog10_emmean']

	# Plot the distributions of MA and MB for each trait
	plot_single_mutants(df, labels)

	# Reshape data to wide format.
	dfw = reshape_data(df, labels)

	# Sort values in MA and MB so that MA is always greater than MB
	dfw = sort_single_mutants(dfw, labels)

	# First, calculate relative fitness
	dfw = calc_relative_fitness(dfw)

	# Calculate epistasis values
	dfw = calc_epistasis(dfw,
		f'{dir_path}/fitness_data_for_Kenia_09232024_{trait}_emmeans_epistasis.tsv')

	del df, dfw, labels
