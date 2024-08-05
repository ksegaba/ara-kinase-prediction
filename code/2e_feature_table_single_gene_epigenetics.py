# Importing packages
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import json

# reading the features csv file
feature_data = pd.read_csv('Feature engineering checklist - 05_Epigenetics.csv')
feature_data.columns

# Reading the gene instance file
instances_dataset_file_path = '/home/seguraab/ara-kinase-prediction/data/instances_dataset_singles.txt'
single_genes = pd.read_csv(instances_dataset_file_path, delimiter='\t', header=0)
single_genes.head()

def apply_transformations(df, feature_name, method):
    """
    Apply transformations based on the feature name suffix and data processing method.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the calculated feature values.
    - feature_name (str): The feature name indicating which transformations to apply.
    - method (str): The data processing method, either 'binary' or 'continuous'.

    Returns:
    - pd.Series: The transformed feature values.
    """
    
    df[feature_name] = df[feature_name].astype(float)
    
    if method == 'Binary':
        # Only apply the _TF transformation if binary
        return df[feature_name]
    else:
        # Apply all transformations if continuous
        if '_log' in feature_name:
            return df[feature_name].apply(lambda x: np.log10(x) if x > 0 else np.nan)
        elif '_reciprocal' in feature_name:
            return df[feature_name].apply(lambda x: 1 / x if x != 0 else np.nan)
        elif '_squared' in feature_name:
            return df[feature_name] ** 2
        elif '_binned' in feature_name:
            return pd.qcut(df[feature_name], 4, labels=False, duplicates='drop')
        else:
            return df[feature_name]
        
# Process single genes
for index, row in tqdm(feature_data.iterrows()):
    file_path = row['File path']
    feature_name = row['ML model feature name']
    method = row['Data processing method']
    
    # Read gene values from the JSON file
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            gene_values = json.load(f)
            
        # Only storing No transformation for binary processing method
        if method == 'Binary' and '_noTF' not in feature_name:
            continue
            
        # Map single gene values
        single_genes[feature_name] = single_genes['gene'].map(gene_values)
            
        # Apply transformations based on the feature name suffix and method
        single_genes[feature_name] = apply_transformations(single_genes, feature_name, method)
    else:
        print('Path not found for the file:', file_path)

        
# Saving the feature matrix/table
single_genes.to_csv("/home/seguraab/ara-kinase-prediction/data/Features/05_epigenetics_single_gene_features.csv", index = False)
        
