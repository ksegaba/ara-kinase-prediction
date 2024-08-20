#!/usr/bin/env python
''' Convert a FASTA file to a PHYLIP file. '''

__author__ = 'Kenia Segura Abá'

import argparse, os, time
import multiprocessing as mp
import datatable as dt
from Bio import AlignIO
from Bio import SeqIO
from Bio.Phylo.PAML import yn00


def parse_args():
	''' Parse command line arguments. '''
	
	parser = argparse.ArgumentParser(description='Convert a FASTA file to a PHYLIP file.')
	
	parser.add_argument('-cds_dir', type=str, help='Path to coding sequence alignment file folder.')
	parser.add_argument('-out_dir', type=str, help='Path to PAML output directory.')
	args = parser.parse_args()
	
	return args


def get_cds_genes(gff):
	''' Get the list of CDS gene identifiers from a GFF file. '''
	
	df = dt.fread(gff, fill=True, skip_to_line=9)
	df = df[dt.f.C2 == 'CDS', 'C8'] # CDS gene descriptions
	df = df.to_pandas()
	genes = df.C8.str.split(';', expand=True)[0].\
			str.replace('ID=cds-', '').unique().tolist()
	
	return genes


def determine_species(records):
	''' Determine which species each gene in an alignment belongs to. '''

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
	pairs = [pair for pair in pairs if len(pair) > 1]
	
	return pairs


def fasta2phylip(prefix, species):
	''' Convert a FASTA file containing a gene pair to a PHYLIP file. '''
	
	os.system(f'java -jar ~/external_software/ALTER/alter-lib/target/ALTER-1.3.4-jar-with-dependencies.jar \
				-i {prefix}_{species}.fasta -if FASTA -ip MUSCLE -io Linux \
				-o {prefix}_{species}.phy -of PHYLIP -op PAML -os -oo Linux')
	
	return f'{prefix}_{species}.phy'


def run_yn00(cds_fasta): # this code works well, but you must use ALTER to convert the fasta file to phylip, not fasta2phylip()
	''' Run PAML yn00 method to calculate dN/dS ratio.
	
	Parameters:
		fasta_file (str): path to input FASTA file
	'''

	# Parse the FASTA file
	alignment = AlignIO.read(cds_fasta, 'fasta')
	records = list(SeqIO.parse(cds_fasta, 'fasta')) 
	len_set = {len(record.seq) for record in records} # sequence lengths

	with open(f'{time.strftime('%Y-%m-%d')}_fasta_files_with_different_lengths.txt', 'a') as f:
		if len(len_set) > 1:
			f.write(f'{fasta_file}\n')
			print(f'Error: {fasta_file} contains sequences of different lengths.\
				{len_set}')
			return None

		else:
			# Identify which gene belongs to which species
			pairs = determine_species(records)
			
			# Convert alignment file to PHYLIP file for pairs of genes
			for pair in pairs:
				pair_recs = [record for record in records if record.id in pair]
				prefix = cds_fasta.replace('.fasta', '')
				# SeqIO.write(pair_recs, f'{prefix}_{pair[0]}.fasta', 'fasta') # write pair to a new file
				# inp = fasta2phylip(prefix, pair[0]) # input to PAML yn00 method (convert to phylip)
				inp = f'{prefix}_{pair[0]}.phy'
				inp_file = inp.split('/')[-1] # extract phylip file name
				
				# Run yn00 method
				yn = yn00.Yn00(alignment=inp, working_dir=out_dir,
						out_file = f"{out_dir}/{inp_file.replace('.phy', '.yn00')}")
				yn.set_options(verbose = True,
					icode = 0,
					weighting = 0,
					commonf3x4 = 0,
					ndata = 1)
				results = yn.run(verbose=True)
				print(results)

		return results


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

	# Convert FASTA files to PHYLIP files
	pool = mp.Pool(mp.cpu_count()) # use all available cores
	pool.map(fasta2phylip, cds_files)
	pool.close()


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
