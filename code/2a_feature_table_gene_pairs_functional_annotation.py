import sys
from sys import argv
import json
import itertools
import pandas as pd
import os
import math

# Function to determine the pairing value
def determine_pair_value(gene1: str, gene2: str , gene_dict: str,action: str = "None") -> float:
    """_summary_

    Args:
        gene1 (str): One of the gene pair
        gene2 (str): another one of the gene pair
        gene_dict (str): The directory of json file where we get gene pairs and their similarity values
        action (str): Support four different type to calculation can be done on the original value

    Returns:
        float: _description_
    """
    in_gene1 = gene_dict.get(gene1, None)
    in_gene2 = gene_dict.get(gene2, None)
    
    value = 0
    
    # the case where values are not numeric or are missing
    try:
        in_gene1 = float(in_gene1)
    except (TypeError, ValueError):
        in_gene1 = math.nan
    try:
        in_gene2 = float(in_gene2)
    except (TypeError, ValueError):
        in_gene2 = math.nan
    
    
    if math.isnan(in_gene1) or math.isnan(in_gene2):
        value = math.nan
    elif in_gene1 == 1 and in_gene2 == 1:
        value = 2
    elif in_gene1 == 1 or in_gene2 == 1:
        value = 1
    else:
        value = 0
    
    if action == 'log':
        Calculated_value = math.log(value) if value > 0 else float('-inf')
    elif action == 'noTF':
        Calculated_value = value
    elif action == 'reciprocal':
        if value ==0 or math.isnan(value):
            Calculated_value = math.nan
        else:
            Calculated_value = 1/value
    else:
        if math.isnan(value):
            Calculated_value = math.nan
        else:
            Calculated_value = value**2
    return Calculated_value

print("Script started")

# Directory containing JSON files
# directory = '/home/tangji19/02_Other_Project/01_Kenia/01_feature_generating/AraCyc_pathways/AraCyc'
directory = '/home/seguraab/ara-kinase-prediction/'
json_dir = 'data/2021_cusack_data/21_arabidopsis_redundancy/01_functional_annotation/AraCyc_pathways/AraCyc'
save_dir = 'data/2021_cusack_data/Dataset_4_Features'

# Ensure the save directory exists
if not os.path.exists(os.path.join(directory, save_dir)):
    os.makedirs(save_dir)

# List of all JSON files in the directory
json_files = [f for f in os.listdir(os.path.join(directory, json_dir)) if f.endswith('.json')]

# Collect all genes from all files to create a comprehensive list
# all_genes = set()
# for file in json_files:
#     with open(os.path.join(directory, file), 'r') as f:
#         gene_dict = json.load(f)
#         all_genes.update(gene_dict.keys())
        
# print(len(all_genes))     

# Generate all possible gene pairs
# gene_pairs = list(itertools.combinations(all_genes, 2))
# print(len(gene_pairs)) 

# Input selected instance txt file
# instance_file = 'instances_dataset_1.txt'
instance_file = 'data/2021_cusack_data/Dataset_4.txt'
file_path = os.path.join(directory, instance_file)
data = pd.read_csv(file_path, sep='\t')
data = data["pair_ID"].str.split("_", expand=True)

# Create a list of tuples from the DataFrame
gene_pairs = list(data.itertuples(index=False, name=None))
print(len(gene_pairs))

# Dataframe to store the results
result_df = pd.DataFrame(gene_pairs, columns=['gene1', 'gene2'])
actions = ["log","noTF","reciprocal","squared"]

# Process each JSON file and generate the respective column in the dataframe
for index, file  in enumerate(json_files):
    # file_path = os.path.join(directory, file)
    file_path = os.path.join(directory, json_dir, file)
    print('json file:', index+1)
    with open(file_path, 'r') as f:
        gene_dict = json.load(f)
        for action in actions:
            column_values = [
            determine_pair_value(pair[0], pair[1], gene_dict,action) for pair in gene_pairs
            ]
            file_name = file.split('_dictionary')[0]
            column_name = f"Continuous_{file_name}_Number_in_pair_{action}"
            result_df[column_name] = column_values
            print(result_df.shape)
            print(result_df.columns)
    # if (index+1)%4 == 0:
    #     file_number = (index+1)//4
    #     report_file_name=f'report_{file_number}.txt'
    #     report_dir = '/home/tangji19/02_Other_Project/01_Kenia/01_feature_generating/AraCyc_pathways/report_data'
    #     report_file_path = os.path.join(report_dir, report_file_name)
    #     result_df.to_csv(report_file_path, sep='\t', index=False)
    #     result_df = pd.DataFrame(gene_pairs, columns=['gene1', 'gene2'])
# Save the result to a text file
# result_df.to_csv('/home/tangji19/02_Other_Project/01_Kenia/01_feature_generating/AraCyc_pathways/gene_pairs.txt', sep='\t', index=False, na_rep='NaN')
result_df.to_csv(os.path.join(directory, save_dir, 'Dataset_4_features_functional_annotations.txt'), sep='\t', index=False, na_rep='NaN')