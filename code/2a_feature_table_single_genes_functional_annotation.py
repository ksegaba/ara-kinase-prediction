import json
import pandas as pd
import os

__author__ = 'Jingyao Tang'

# Directory containing JSON files
directory = '/home/tangji19/02_Other_Project/01_Kenia/01_feature_generating/AraCyc_pathways/AraCyc'

# List of all JSON files in the directory
json_files = [f for f in os.listdir(directory) if f.endswith('.json')]

# Process each JSON file and generate the respective column in the dataframe
for index, file  in enumerate(json_files):
    file_path = os.path.join(directory, file)
    column_name = file.split('_dictionary')[0]
    # print('json file:', index+1)
    with open(file_path, 'r') as f:
        gene_dict = json.load(f)
        if index ==0:
            single_gene_value = pd.DataFrame.from_dict(gene_dict, orient='index', columns=[column_name])
        else:
            add_columns = pd.DataFrame.from_dict(gene_dict, orient='index', columns=[column_name])
            single_gene_value = pd.merge(single_gene_value, add_columns, left_index=True, right_index=True, how='outer')
print(single_gene_value.shape)
# Save the result to a text file
single_gene_value.to_csv('/home/tangji19/02_Other_Project/01_Kenia/01_feature_generating/AraCyc_pathways/Single_gene_features.txt', sep='\t', index=True, na_rep='NaN')
