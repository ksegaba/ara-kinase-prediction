# Importing packages
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import json

os.chdir('/home/seguraab/ara-kinase-prediction')

# reading the features csv file
# feature_data = pd.read_csv('Feature engineering checklist - 05_Epigenetics.csv')
feature_data = pd.read_csv('data/Features/05_epigenetics_feature_list.csv')
feature_data.columns

# Reading the gene_pair files
# instances_dataset_1_file_path = '/home/seguraab/ara-kinase-prediction/data/instances_dataset_1.txt'
instances_dataset_1_file_path = '/home/seguraab/ara-kinase-prediction/data/2021_cusack_data/Dataset_4.txt'
gene_pairs = pd.read_csv(instances_dataset_1_file_path, delimiter='\t', header=0)
gene_pairs = gene_pairs["pair_ID"].str.split("_", expand=True)
gene_pairs.columns = ['gene1', 'gene2']
gene_pairs.head()


def calculate_feature_value(gene1, gene2, gene_values, operation):
    """
    Calculate the feature value for a pair of genes based on the specified operation.

    Parameters:
    - gene1 (str): The first gene identifier.
    - gene2 (str): The second gene identifier.
    - gene_values (dict): A dictionary containing gene values.
    - operation (str): The operation to perform for calculating the feature value. One of:
      ['Number in pair', 'Average', 'Difference (absolute value)', 'Maximum', 'Minimum', 'Pair total']

    Returns:
    - float: The calculated feature value, or np.nan if either gene value is missing or non-numeric.
    """
    
    value1 = gene_values.get(gene1, np.nan)
    value2 = gene_values.get(gene2, np.nan)
    
    # the case where values are not numeric or are missing
    try:
        value1 = float(value1)
    except (TypeError, ValueError):
        value1 = np.nan
    try:
        value2 = float(value2)
    except (TypeError, ValueError):
        value2 = np.nan
    
    if np.isnan(value1) or np.isnan(value2):
        return np.nan
    
    if operation in ['Number in pair', 'Pair total']:
        return value1 + value2
    elif operation == 'Average':
        return (value1 + value2) / 2
    elif operation == 'Difference (absolute value)':
        return abs(value1 - value2)
    elif operation == 'Maximum':
        return max(value1, value2)
    elif operation == 'Minimum':
        return min(value1, value2)
    else:
        return np.nan
    
    
def apply_transformations(df, feature_name):
    """
    Apply transformations based on the feature name suffix.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the calculated feature values.
    - feature_name (str): The feature name indicating which transformations to apply.

    Returns:
    - pd.Series: The transformed feature values.
    """
    
    if '_log' in feature_name:
        return df[feature_name].apply(lambda x: np.log10(x) if x > 0 else np.nan)
    elif '_reciprocal' in feature_name:
        return df[feature_name].apply(lambda x: 1 / x if x != 0 else np.nan)
    elif '_squared' in feature_name:
        return df[feature_name] ** 2
    elif '_binned' in feature_name:
        # dropping duplicates as there is error indicates that there are duplicate bin edges, which can happen if the data has many identical values, especially zeros
        return pd.qcut(df[feature_name], 4, labels=False, duplicates='drop')
    else:
        return df[feature_name]
    
    
# Process each file path and update the gene_pairs DataFrame
for index, row in tqdm(feature_data.iterrows()):
    file_path = row['File path']
    feature_name = row['ML model feature name']
    operation = row['Calculation for gene pair']
    
    # Read gene values from the JSON file
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            gene_values = json.load(f)
            
        # gene pair value
        gene_pairs[feature_name] = gene_pairs.apply(
                lambda row: calculate_feature_value(row['gene1'], row['gene2'], gene_values, operation), axis=1
            )
        
        # Apply transformations based on the feature name suffix
        gene_pairs[feature_name] = apply_transformations(gene_pairs, feature_name)
            
    else:
        print('Path not found for the file :', file_path)


# Saving the feature matrix/table
# gene_pairs.to_csv("/home/seguraab/ara-kinase-prediction/data/Features/05_epigenetics_gene_pair_features.csv", index = False)
gene_pairs.to_csv("data/2021_cusack_data/Dataset_4_Features/Dataset_4_features_epigenetics.txt", sep='\t', index=False)
