#!/usr/bin/env python
'''This script implements PAML yn00 method to calculate the dN, dS, and dN/dS 
ratio for each Arabidopsis thaliana gene.

Parameters:
	-cds_dir: Path to coding sequence alignment file folder.
	-out_dir: Path to PAML output directory.

Returns:
	- A JSON file containing the results of the PAML yn00 method.
	- A log file containing any errors encountered during execution.
	'''

__author__ = 'Kenia Segura Abá'

import argparse, os, time, json, Bio, tqdm
import multiprocessing as mp
import datatable as dt
import pandas as pd
from Bio import AlignIO
from Bio import SeqIO
from Bio.Phylo.PAML import yn00


def parse_args():
	'''Parse command line arguments.'''
	
	parser = argparse.ArgumentParser(description='Convert a FASTA file to a PHYLIP file.')
	
	parser.add_argument('-cds_dir', type=str, help='Path to coding sequence alignment file folder.')
	parser.add_argument('-out_dir', type=str, help='Path to PAML output directory.')
	args = parser.parse_args()
	
	return args


def get_cds_genes(gff):
	'''Get the list of CDS gene identifiers from a GFF file.'''
	
	df = dt.fread(gff, fill=True, skip_to_line=9)
	df = df[dt.f.C2 == 'CDS', 'C8'] # CDS gene descriptions
	df = df.to_pandas()
	genes = df.C8.str.split(';', expand=True)[0].\
			str.replace('ID=cds-', '').unique().tolist()
	
	return genes


def determine_species(records):
	'''Determine which species each gene in an alignment belongs to.'''

	pairs = [['ptr'], ['gma'], ['sly'], ['tca']]
	for record in records:
		if record.id in ath_genes: # A. thaliana
			for i in range(4):
				pairs[i].append(record.id)
		elif (record.id in ptr_genes) and (record.id is not None): # P. trichocarpa
			pairs[0].append(record.id)
		elif (record.id in gma_genes and record.id is not None): # G. max
			pairs[1].append(record.id)
		elif (record.id in sly_genes and record.id is not None): # S. lycopersicum
			pairs[2].append(record.id)
		elif (record.id in tca_genes and record.id is not None): # T. cacao
			pairs[3].append(record.id)
	
	# Remove pairs with less than two genes
	pairs = [pair for pair in pairs if len(pair) > 2]
	
	return pairs


def fasta2phylip(prefix):
	'''Convert a FASTA file to a PHYLIP file.'''
	
	os.system(f'java -jar ~/external_software/ALTER/alter-lib/target/ALTER-1.3.4-jar-with-dependencies.jar \
				-i {prefix}.fasta -if FASTA -ip MUSCLE -io Linux \
				-o {prefix}.phy -of PHYLIP -op PAML -os -oo Linux')
	
	return f'{prefix}.phy'


def run_yn00(cds_fasta): # this code works well, but you must use ALTER to convert the fasta file to phylip, not fasta2phylip()
	''' Run PAML yn00 method to calculate dN, dS, and the dN/dS ratio.
	
	Parameters:
		fasta_file (str): path to input FASTA file
	'''

	# Parse the FASTA file
	records = list(SeqIO.parse(cds_fasta, 'fasta'))
	len_set = {len(record.seq) for record in records} # sequence lengths

	if len(len_set) > 1:
		with open(f"{time.strftime('%Y-%m-%d')}_fasta_files_with_different_lengths.txt", 'a') as f:
			f.write(f'{cds_fasta}\n')
			print(f'Error: {cds_fasta} contains sequences of different lengths.\
				{len_set}')
		return {}

	else:
		# Identify which gene belongs to which species
		pairs = determine_species(records) # gene pairs
		res_dict = {} # PAML results dictionary
		for pair in pairs:
			# Convert alignment file to PHYLIP file for pairs of genes
			pair_recs = [record for record in records if record.id in pair]
			prefix = cds_fasta.replace('.fasta', '')
			SeqIO.write(pair_recs, f'{prefix}_{pair[0]}.fasta', 'fasta') # write pair to a new file
			inp = fasta2phylip(f'{prefix}_{pair[0]}') # input to PAML yn00 method (convert to phylip)
			inp_file = inp.split('/')[-1] # extract phylip file name
			
			# Run yn00 method
			yn = yn00.Yn00(alignment=inp, working_dir=out_dir,
					out_file = f"{out_dir}/{inp_file.replace('.phy', '.yn00')}")
			yn.set_options(verbose = True,
				icode = 0,
				weighting = 0,
				commonf3x4 = 0,
				ndata = 1)
			
			try:
				results = yn.run(verbose=True)

				# Get dN and dS values
				gene1_key = next(iter(results))
				gene2_key = next(iter(results[gene1_key]))
				data = pd.DataFrame.from_dict(results[gene1_key][gene2_key]).T[['dN', 'dS']]
				ratio = data['dN'] / data['dS']

				# Determine which gene is the Arabidopsis gene and save results
				if gene1_key in ath_genes:
					if gene1_key not in res_dict.keys():
						res_dict[gene1_key] = {}
						res_dict[gene1_key]['dN'] = {gene2_key:data['dN'].to_dict()}
						res_dict[gene1_key]['dS'] = {gene2_key:data['dS'].to_dict()}
						res_dict[gene1_key]['dN/dS'] = {gene2_key:ratio.to_dict()}
					else:
						res_dict[gene1_key]['dN'][gene2_key] = data['dN'].to_dict()
						res_dict[gene1_key]['dS'][gene2_key] = data['dS'].to_dict()
						res_dict[gene1_key]['dN/dS'][gene2_key] = ratio.to_dict()
					
				if gene2_key in ath_genes:
					if gene2_key not in res_dict.keys():
						res_dict[gene2_key] = {}
						res_dict[gene2_key]['dN'] = {gene1_key:data['dN'].to_dict()}
						res_dict[gene2_key]['dS'] = {gene1_key:data['dS'].to_dict()}
						res_dict[gene2_key]['dN/dS'] = {gene1_key:ratio.to_dict()}
					else:
						res_dict[gene2_key]['dN'][gene1_key] = data['dN'].to_dict()
						res_dict[gene2_key]['dS'][gene1_key] = data['dS'].to_dict()
						res_dict[gene2_key]['dN/dS'][gene1_key] = ratio.to_dict()

			except FileNotFoundError as e:
				with open(f"{out_dir}/{time.strftime('%Y-%m-%d')}_yn00_error_log.txt", 'a') as f:
					f.write(f'FileNotFoundError: {inp}\n')
			except Bio.Phylo.PAML._paml.PamlError as e:
				with open(f"{out_dir}/{time.strftime('%Y-%m-%d')}_yn00_error_log.txt", 'a') as f:
					f.write(f'PamlError: {inp}\n')
			except:
				with open(f"{out_dir}/{time.strftime('%Y-%m-%d')}_yn00_error_log.txt", 'a') as f:
					f.write(f'General error: {inp}\n')
				continue
		
		return res_dict


def worker(input_data):
	'''This function must be adapted to accept the required input data and 
	call run_yn00() appropriately.'''
	
	res_dict = run_yn00(input_data)

	return res_dict


def aggregate_results(input_list):
	'''Combine yn00 results dictionaries into a JSON file.'''	

	# Create a processing pool
	with mp.Pool(processes=mp.cpu_count()) as pool:
		# Execute 'worker' for each element in input_list
		pool_res = pool.map(worker, input_list)
	
	# Combine results
	combined_results = {}
	for res_dct in pool_res:
		if len(res_dct) > 0:
			combined_results.update(res_dct)
		else:
			continue
	
	# Save the dictionary into a JSON file
	with open(f'{out_dir}/arabidopsis_paml_combined_results.json', 'w') as f:
		json.dump(combined_results, f)


if __name__ == '__main__':
	# Parse command line arguments
	args = parse_args()
	cds_dir = args.cds_dir
	out_dir = args.out_dir

	# Gene lists for each species
	ath_genes = get_cds_genes('~/ara-kinase-prediction/data/NCBI_genomes/GCF_000001735.4/genomic.gff')
	ptr_genes = get_cds_genes('~/ara-kinase-prediction/data/NCBI_genomes/GCF_000002775.5/genomic.gff')
	gma_genes = get_cds_genes('~/ara-kinase-prediction/data/NCBI_genomes/GCF_000004515.6/genomic.gff')
	sly_genes = get_cds_genes('~/ara-kinase-prediction/data/NCBI_genomes/GCF_000188115.5/genomic.gff')
	tca_genes = get_cds_genes('~/ara-kinase-prediction/data/NCBI_genomes/GCF_000208745.1/genomic.gff')

	# Get list of FASTA files
	os.chdir(cds_dir)
	cds_files = [f'{cds_dir}/{f}' for f in os.listdir('.') if f.endswith('_cds_aligned.fasta')]
	
	# Run yn00 method in parallel
	# aggregate_results(cds_files)
	'''I cannot run this code in parallel. Sometimes the yn.run() method fails 
	because the worker does not detect the .fasta or .phy files that were created.
	This tells me the workers aren't getting the data they need even though it 
	exists. fasta2phylip seems to be working fine, it's the yn00 call that is 
	messed up. To avoid this, I will run the code sequentially.'''
	
	# Run yn00 method for each file sequentially
	combined_results = {}
	for file in tqdm.tqdm(cds_files):
		res_dict = run_yn00(file)
		if len(res_dict) > 0:
			combined_results.update(res_dict)
			
	# Save the dictionary into a JSON file
	with open(f'{out_dir}/arabidopsis_paml_combined_results.json', 'w') as f:
		json.dump(combined_results, f)

### GRAVEYARD
# def fasta2phylip(fasta_file):
# 	''' Convert a FASTA file to a PHYLIP file
	
# 	Parameters:
# 		fasta_file (str): path to input FASTA file
# 	'''
	
# 	# Parse FASTA file
# 	alignment = AlignIO.read(fasta_file, 'fasta')
# 	records = list(SeqIO.parse(fasta_file, 'fasta'))
# 	len_set = {len(record.seq) for record in records} # sequence lengths
	
# 	# Get the list of fasta files that contained sequences of different lengths
# 	with open(f'{time.strftime('%Y-%m-%d')}_fasta_files_with_different_lengths.txt', 'a') as f:
# 		if len(len_set) > 1:
# 			f.write(f'{fasta_file}\n')
# 			print(f'Error: {fasta_file} contains sequences of different lengths.\
# 				{len_set}')
# 			return None

# 		else:
# 			# SeqIO.write(records, fasta_file + '.phylip', 'phylip-sequential') # Not the correct format for PAML, used ALTER instead
# 			SeqIO.write(records, fasta_file + '.phylip', 'phylip') # also not correct
# 			return fasta_file + '.phylip'
