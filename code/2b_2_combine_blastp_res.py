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

# Combine the identifiers of the top gene hits
combined = pd.concat([gm_res[1].str.split(' ', expand=True)[0],
           Tc_res[1].str.split(' ', expand=True)[0],
           Pt_res[1].str.split(' ', expand=True)[0],
           Sl_res[1].str.split(' ', expand=True)[0]],
          axis=1, ignore_index=False)
combined.columns = ['Gmax', 'Tcacao', 'Ptrichocarpa', 'Slycopersicum']
combined.index.name = 'Athaliana'
combined.fillna('NA', inplace=True)
combined.to_csv(f'{path}/blastp_TAIR10_top_hits_all.txt', sep='\t')