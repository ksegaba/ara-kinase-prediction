#!/usr/bin/env python3
'''Combine feature tables from the different data categories and save as a 
single file'''
# CONVERT THIS FILE TO IPYNB

import os
import datatable as dt
import pandas as pd
import numpy as np

################## INTEGRATE FEATURE TABLES BEFORE IMPUTATION ##################
os.chdir('/home/seguraab/ara-kinase-prediction/data/Features')

# Load in all the feature tables
epi_df = dt.fread('epigenetics_gene_pairs_features.csv').to_pandas().\
    set_index(['gene1', 'gene2'])
evo_df = dt.fread('evolutionary_properties_gene_pairs_features.txt').\
    to_pandas().set_index(['gene1', 'gene2'])
expr_df = dt.fread('gene_expression_gene_pairs_features.csv').to_pandas().\
    set_index(['gene1', 'gene2'])
func_df = dt.fread('functional_annotations_gene_pairs_features.txt').\
    to_pandas().set_index(['gene1', 'gene2'])
net_df = dt.fread('network_properties_gene_pairs_features.csv').to_pandas().\
    set_index(['gene1', 'gene2'])
prot_files = [f for f in os.listdir('.') if f.startswith('protein_properties')]
prot_df = pd.concat(
    [dt.fread(f).to_pandas().set_index(['gene1', 'gene2']) for f in prot_files],
    ignore_index=False, axis=1)

# Drop columns with all NaN values
func_df = func_df.dropna(axis=1, how='all') # will have to check on this later, the table code doesn't seem to be wrong, but perhaps many were dropped because I'm not looking at the whole genome.
prot_df = prot_df.dropna(axis=1, how='all')

# Combine all the feature tables
all = pd.concat([epi_df, evo_df, expr_df, func_df, net_df, prot_df], axis=1, 
    ignore_index=False)
all.shape # (10250, 4791)

# Separate the training and test sets
instances = pd.read_csv(
    '../20240725_melissa_ara_data/interactions_fitness.csv') # training instances
instances.MA = instances.MA.str.upper()
instances.MB = instances.MB.str.upper()
instances.index = np.sort(instances[['MA', 'MB']], axis=1) # sort the gene pairs
instances.index = instances.index.map(tuple) # convert to tuples
instances.index = instances.index.map(lambda x: x[0] + '_' + x[1]) # convert to strings
instances = instances.loc[:,'Interaction']
# instances.to_csv('../20240725_melissa_ara_data/interactions_fitness_labels.csv')

train = all.loc[instances.index]
test = all.drop(index=instances.index)

# How much missing data is there in the training and test sets?
train.isna().sum() / train.shape[0] * 100
test.isna().sum() / test.shape[0] * 100
train.shape # (126, 4791)
test.shape # (10125, 4791)
sum(test.index.isin(train.index)) #  0; don't know why 126 + 10125 != 10250

# What's remaining if I drop all cols with missing data?
test_no_na = test.dropna(axis=1, how='any')
train_no_na = train.loc[:, test_no_na.columns]
test_no_na.shape # (10125, 1204)
train_no_na.shape # (126, 1204)
test_no_na.isna().sum().sum() # 0
train_no_na.isna().sum().sum() # 0
tmp = train.dropna(axis=1, how='any') # I don't have many training instances, so I can use this to justify dropping columns for now.
tmp.shape # (126, 1204); same as train_no_na
test.loc[:, tmp.columns].shape # (10125, 1204); same as test_no_na
sum(test_no_na.columns==test.loc[:, tmp.columns].columns) # 1204
tmp.isna().sum().sum() # 0
test.loc[:, tmp.columns].isna().sum().sum() # 0

# Save the features that were dropped and those that were kept in two files
pd.Series(tmp.columns).to_csv('features_kept_kinase_prediction.csv', index=False, header=False)
pd.Series([c for c in all.columns if c not in tmp.columns]).\
    to_csv('features_dropped_kinase_prediction.csv', index=False, header=False)

# Write these preliminary feature table to a file
to_save = all.loc[:, tmp.columns]
to_save.index = ['_'.join(i) for i in to_save.index]
to_save.index.name = 'pair'
to_save.to_csv('Table_features_kept_kinase_prediction.csv')
tmp.index = ['_'.join(i) for i in tmp.index]
tmp.index.name = 'pair'
tmp.insert(0, 'Y', instances.Interaction.values)

# I did the following to solve errors I was getting with the original 126x1204 matrix
tmp.drop_duplicates(keep='first', inplace=True) # At1g18620	At1g74160 is duplicated in Melissa's file
cols_with_inf = tmp.columns.to_series()[np.isinf(tmp).any()]
tmp_no_inf = tmp.drop(columns=cols_with_inf)
tmp_no_inf.to_csv('Table_features_kept_kinase_prediction_train.csv')

# Save the test set instances into a file
test.index = ['_'.join(i) for i in test.index]
pd.Series(test.index).to_csv('Test_instances_kinase_prediction.csv', index=False, header=False)

'''Update September 11, 2024
The binary classification models I built so far are not learning the positive 
set at all. It could be because the features for training and test sets are very
similar. It can also be because my data is imbalanced. The third reason is that
I don't have enough training samples. Lastly, perhaps XGBoost is too complicated
of a model to use on this dataset.'''

# What is the correlation of features between training and test sets?
os.chdir('/home/seguraab/ara-kinase-prediction')
X = pd.read_csv('data/Features/Table_features_kept_kinase_prediction_train.csv', index_col=0)
X_corr = X.T.corr(method='pearson') # correlation between instances across feature values
test_files = [f for f in os.listdir('data/') if f.startswith('test_ara_m_')]

# assign the X_corr index values to the corresponding test file. The test files contain subsets of the X_corr index values.
X_corr['Test_file'] = ''
for f in test_files:
    test = pd.read_csv('data/' + f, header=None)
    test_corr = X_corr.loc[test[0], test[0]]
    X_corr.loc[test[0], 'Test_file'] = f

# map the test file names to RGB colors
X_corr['Test_file_color'] = X_corr.Test_file.apply(lambda x: sns.color_palette('tab20', n_colors=10)[int(x.split('_')[-1][:-4])])
color_dict = dict(zip(X_corr.Test_file, X_corr.Test_file_color))

# Create a clustermap of X_corr and add the test file names as a colored 
# rectangles on the rows and columns of the clustermap
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
g = sns.clustermap(X_corr.drop(columns=['Test_file', 'Test_file_color']),
                   cmap='RdBu_r', center=0, method='average',
                   row_colors=X_corr.Test_file_color,
                   col_colors=X_corr.Test_file_color, xticklabels=False,
                   yticklabels=False,)

# add a legend with the test file colors
legend_patches = [mpatches.Patch(color=color, label=label) for label, color in color_dict.items()]
g.fig.legend(handles=legend_patches, loc='lower left', bbox_to_anchor=(0, 0),
           fontsize='small', title='Test file')

plt.savefig('data/Features/Table_features_kept_kinase_prediction_train_clustermap.pdf')
plt.close()

# Now create a similar figure, but sort the rows and columns by the test file
X_corr_sorted = X_corr.sort_values('Test_file')
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
g = sns.heatmap(X_corr_sorted.drop(columns=['Test_file', 'Test_file_color']),
                cmap='RdBu_r', center=0, xticklabels=False, yticklabels=False)

# add a legend with the test file colors
legend_patches = [mpatches.Patch(color=color, label=label) for label, color in color_dict.items()]
plt.legend(handles=legend_patches, loc='lower left', bbox_to_anchor=(0, 0),
           fontsize='small', title='Test file')

# add the test file colors as rectangles on the rows and columns
space = 0.0005 # space between rectangles
for i in range(len(X_corr_sorted)):
    ax.add_patch(plt.Rectangle(xy=(i, -1), width=1-space, height=1, transform=g.transData, clip_on=False, color=X_corr_sorted.Test_file_color.iloc[i]))
    ax.add_patch(plt.Rectangle(xy=(-1, i), width=1, height=1-space, transform=g.transData, clip_on=False, color=X_corr_sorted.Test_file_color.iloc[i]))
    # ax.add_patch(plt.Rectangle(xy=(i - space/2, -1 - space/2), width=1+space, height=1, transform=g.transData, clip_on=False, color=X_corr_sorted.Test_file_color.iloc[i]))
    # ax.add_patch(plt.Rectangle(xy=(-1 - space/2, i - space/2), width=1, height=1-space, transform=g.transData, clip_on=False, color=X_corr_sorted.Test_file_color.iloc[i]))

plt.savefig('data/Features/Table_features_kept_kinase_prediction_train_heatmap.pdf')
plt.close()

####################### INTEGRATE IMPUTED FEATURE TABLES #######################
os.chdir('imputed_features')
instances = pd.read_csv(
    '../../20240725_melissa_ara_data/interactions_fitness_labels.csv')
instances.drop_duplicates(keep='first', inplace=True)
instances.set_index('Unnamed: 0', inplace=True)

for i in range(10):
    # Get the list of files for each train and test set
    train_files = [f for f in os.listdir('.') if f.endswith(str(i) + '_imputed_train.csv')]
    test_files = [f for f in os.listdir('.') if f.endswith(str(i) + '_imputed_test.csv')]
    train_files.sort()
    test_files.sort()
    
    # Load in the feature tables
    df_train = []
    df_test  = []
    for j in range(len(train_files)):
        train = pd.read_csv(train_files[j], index_col=0)
        test = pd.read_csv(test_files[j], index_col=0)
        df_train.append(train)
        df_test.append(test)
        del train, test
    
    # Combine and save
    combined = pd.concat([pd.concat(df_train, axis=1, ignore_index=False),
        pd.concat(df_test, axis=1, ignore_index=False)], axis=0)
    combined.insert(0, 'Y', instances.loc[combined.index,:].values)
    
    combined.to_csv(
        'Table_features_imputed_kinase_prediction_train_test_ara_m_fold_' + str(i) + '.csv')
    del combined, train_files, test_files, df_train, df_test