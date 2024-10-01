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

# Prepare corrected fitness data
data = dt.fread(
  '../data/20240917_melissa_ara_data/corrected_data/fitness_data_for_Kenia_09172024_corrected.tsv').\
  to_pandas()
  
data['MA'] = 0
data.loc[data.Genotype == 'MA', 'MA'] = 1
data['MB'] = 0
data.loc[data.Genotype == 'MB', 'MB'] = 1
data['DM'] = 0
data.loc[data.Genotype == 'DM', 'DM'] = 1
data.loc[data.Genotype == 'DM', 'MA'] = 1
data.loc[data.Genotype == 'DM', 'MB'] = 1

# %%
# Plot the distributions of MA and MB for each trait
for label in [l for l in data.columns if '_corrected' in l]:
	if label=='TSC_corrected':
		MA_vals = data.loc[data.Genotype == 'MA', label]
		MB_vals = data.loc[data.Genotype == 'MB', label]
		fig = plt.figure()
		sns.histplot(MA_vals, color='blue', label='MA')
		sns.histplot(MB_vals, color='red', label='MB')
		plt.title(label)
		plt.legend()
		plt.show()

# %%

'''Create all possible pairs of MA and MB rows, then sort the values so that MA 
is always greater than MB. Calculate the relative fitness, expected DM fitness,
and epistasis values for the four established genetic interaction definitions'''

for label in [l for l in data.columns if '_corrected' in l]:
    MA_vals = data.loc[data.Genotype == 'MA', label].unique()
    MB_vals = data.loc[data.Genotype == 'MB', label].unique()
    DM_vals = data.loc[data.Genotype == 'DM', label].unique()
    WT_vals = data.loc[data.Genotype == 'WT', label].unique()
    
    all_pairs = list(itertools.product(MA_vals, MB_vals)) # all possible pairs
    all_pairs = [sorted(pair, reverse=True) for pair in all_pairs] # sort pairs
    
    # add the DM and WT values to the pairs
    all_pairs = list(itertools.product(MA_vals, MB_vals, DM_vals, WT_vals))

# %%
'''Calculate epistasis values based on 4 established genetic interaction
definitions (Mani 2008) using the estimated marginal means of each label:
epistasis = W_ab - Expected(W_ab)
- Minimum: epistasis = W_ab - min(W_a, W_b)
- Product: epistasis = W_ab - W_a * W_b
- Additive: epistasis = W_ab - (W_a + W_b - 1)
- Log-additive: epistasis = W_ab - np.log2((2**W_a - 1) * (2**W_b - 1) + 1)'''

# Reshape data to wide format.
data = data.loc[:, ['Set', 'Genotype', 'GN_emmean', 'PG_emmean', 'DTB_emmean',
                    'LN_emmean', 'DTF_emmean', 'SN_emmean', 'WO_emmean',
                    'FN_emmean', 'SPF_emmean', 'TSC_emmean', 'SH_emmean']]
data.drop_duplicates(inplace=True)

## check each set has 4 genotypes
tmp = data.groupby('Set')['Genotype'].value_counts().reset_index('Genotype')
sum(tmp.groupby('Set')['Genotype'].nunique() == 4) == tmp.index.nunique() # True
del tmp

dataw = data.pivot_table(index='Set', columns='Genotype',
  values=['GN_emmean', 'PG_emmean', 'DTB_emmean', 'LN_emmean', 'DTF_emmean',
  'SN_emmean', 'WO_emmean', 'FN_emmean', 'SPF_emmean', 'TSC_emmean', 'SH_emmean'])
dataw.shape # (127, 44)

# Sort values in MA and MB so that MA is always greater than MB
for trait in dataw.columns.levels[0]:
  for row in dataw.index:
    if dataw.loc[row, (trait, 'MA')] < dataw.loc[row, (trait, 'MB')]:
      tmp_MA = dataw.loc[row, (trait, 'MA')]
      dataw.loc[row, (trait, 'MA')] = dataw.loc[row, (trait, 'MB')]
      dataw.loc[row, (trait, 'MB')] = tmp_MA

# First, calculate relative fitness
# note, some sets have WT = 0, so the relative fitness will be undefined.
for trait in dataw.columns.levels[0]:
  dataw.loc[:,trait] = dataw[trait].apply(lambda x: x / x['WT'], axis=1).values

# Add W_ to column names to denote relative fitness
dataw.columns = pd.MultiIndex.from_arrays(
  [dataw.columns.get_level_values(0).values,
   'W_' + dataw.columns.get_level_values(1).values])

# Calculate epistasis values
for trait in dataw.columns.levels[0]:
  dataw[(trait, 'epi_min')] = dataw[trait].apply(lambda x: x['W_DM'] - min(x['W_MA'], x['W_MB']), axis=1).values
  dataw[(trait, 'epi_product')] = dataw[trait].apply(lambda x: x['W_DM'] - (x['W_MA'] * x['W_MB']), axis=1).values
  dataw[(trait, 'epi_additive')] = dataw[trait].apply(lambda x: x['W_DM'] - (x['W_MA'] + x['W_MB'] - 1), axis=1).values
  dataw[(trait, 'epi_log_mani')] = dataw[trait].apply(lambda x: x['W_DM'] - np.log2((2**x['W_MA'] - 1) * (2**x['W_MB'] - 1) + 1), axis=1).values
  dataw[(trait, 'epi_mean')] = dataw[trait].apply(lambda x: np.mean([x['W_MA'], x['W_MB']]), axis=1).values
  dataw[(trait, 'epi_max')] = dataw[trait].apply(lambda x: x['W_DM'] - max(x['W_MA'], x['W_MB']), axis=1).values
  dataw[(trait, 'epi_log_additive')] = dataw[trait].apply(lambda x: x['W_DM'] - np.log(x['W_MA'] + x['W_MB']), axis=1).values
  dataw[(trait, 'epi_log_difference')] = dataw[trait].apply(lambda x: x['W_DM'] - np.log(x['W_MA'] - x['W_MB']), axis=1).values

dataw.to_csv('../data/20240917_melissa_ara_data/corrected_data/fitness_data_for_Kenia_09172024_corrected_epistasis_emm.csv')

# # Calculate the standard deviation from the standard error for a trait estimated means value
# n = data.groupby('Set')['Genotype'].value_counts() # number of replicates in each set, per genotype
# SEs = data[['Set', 'Genotype'] + [col for col in data.columns if '_SE' in col]]
# SEs.set_index(['Set', 'Genotype'], inplace=True)
# SEs.drop_duplicates(inplace=True)
# SDs = SEs.apply(lambda x: x * np.sqrt(n.loc[x.name[0], x.name[1]]), axis=1) # standard deviation
# SDs = SDs.pivot_table(index='Set', columns='Genotype', values=SDs.columns)
# SDs.columns = pd.MultiIndex.from_arrays(
#   [SDs.columns.get_level_values(0).str.replace('_SE', '_SD'),
#    SDs.columns.get_level_values(1).values]) # rename columns to denote SD

# # Calculate a p-value that compares W_DM to Expected(W_DM)
# sum(n>30) == len(n) # True, all have more than 30 samples, so we can use a z-test
# for trait in dataw.columns.levels[0]:
#   diff = dataw[(trait, 'W_DM')] - dataw[(trait, 'W_DM')]

# # Rows and columns with undefined values
# with_na = dataw.loc[dataw.isna().any(axis=1)]
# with_na.loc[:, with_na.isna().any()]

# %%
