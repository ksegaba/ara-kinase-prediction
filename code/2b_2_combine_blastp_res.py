#!/usr/bin/env python3
'''
For each Arabidopsis thaliana gene, combine the identifiers of the top gene 
hits from the BLASTp results into one file. One set of identifiers per line.
'''

import pandas as pd

# Read in the BLASTp results
path = '/home/seguraab/ara-kinase-prediction/data/evolutionary_properties_data/0_blast_res'

gm_res = pd.read_csv(f'{path}/TAIR10_Gmax_blastp.txt', index_col=0, sep='\t', header=None)
Tc_res = pd.read_csv(f'{path}/TAIR10_Tcacao_blastp.txt', index_col=0, sep='\t', header=None)
Pt_res = pd.read_csv(f'{path}/TAIR10_Ptrichocarpa_blastp.txt', index_col=0, sep='\t', header=None)
Sl_res = pd.read_csv(f'{path}/TAIR10_Slycopersicum_blastp.txt', index_col=0, sep='\t', header=None)

# Combine the identifiers of the top gene hits for individual arabidopsis genes
combined = pd.concat([gm_res[1].str.split(' ', expand=True)[0],
           Tc_res[1].str.split(' ', expand=True)[0],
           Pt_res[1].str.split(' ', expand=True)[0],
           Sl_res[1].str.split(' ', expand=True)[0]],
          axis=1, ignore_index=False)
combined.columns = ['Gmax', 'Tcacao', 'Ptrichocarpa', 'Slycopersicum']
combined.index.name = 'Athaliana'
combined.fillna('NA', inplace=True)
combined.to_csv(f'{path}/blastp_TAIR10_top_hits_all.txt', sep='\t')

### Combine the IDs for pairs of Arabidopsis genes along with their top hits ###
single_res = pd.read_csv(f'{path}/blastp_TAIR10_top_hits_all.txt', sep='\t') # gene and top hits

instances = pd.read_csv('~/ara-kinase-prediction/data/instances_dataset_1.txt', sep='\t') # gene pairs of interest

id_map = pd.read_csv('~/ara-kinase-prediction/data/NCBI_genomes/GCF_000001735.4/TAIR10_NCBI_REFSEQ_mapping_PROT',
                     sep='\t', names=['Num_ID', 'NCBI_ID', 'TAIR10_ID']) # NCBI identifier mapping
id_map = id_map[['NCBI_ID', 'TAIR10_ID']].set_index('NCBI_ID').to_dict() # dictionary of NCBI to TAIR10 IDs

# Map the NCBI IDs to TAIR10 IDs
single_res['Athaliana'] = single_res['Athaliana'].apply(lambda x: id_map[x] if x in id_map.keys() else x)
                                                        #map(id_map['TAIR10_ID'])

# Combine the top hits for pairs of Arabidopsis genes
for gene1, gene2 in instances[['gene1', 'gene2']].values:
    gene1_res = single_res.loc[single_res['Athaliana'].str.contains(gene1)]
    gene2_res = single_res.loc[single_res['Athaliana'].str.contains(gene2)]
    combined_res = pd.concat([gene1_res, gene2_res], axis=0)
    combined_res.to_csv(f'{path}/blastp_TAIR10_top_hits_{gene1}_{gene2}.txt', sep='\t')