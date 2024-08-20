#!/usr/bin/env python
''' Convert a FASTA file to a PHYLIP file. '''

__author__ = 'Kenia Segura Abá'

import argparse, os, time
import multiprocessing as mp
from Bio import SeqIO
from Bio import AlignIO
from Bio.Phylo.PAML import yn00

def parse_args():
	''' Parse command line arguments. '''
	
	parser = argparse.ArgumentParser(description='Convert a FASTA file to a PHYLIP file.')
	
	parser.add_argument('-cds_dir', type=str, help='Path to coding sequence alignment file folder.')
	parser.add_argument('-out_dir', type=str, help='Path to PAML output directory.')

	args = parser.parse_args()
	
	return args


def fasta2phylip(fasta_file):
	''' Convert a FASTA file to a PHYLIP file
	
	Parameters:
		fasta_file (str): path to input FASTA file
	'''
	
	# Parse FASTA file
	alignment = AlignIO.read(fasta_file, 'fasta')
	records = list(SeqIO.parse(fasta_file, 'fasta'))
	len_set = {len(record.seq) for record in records} # sequence lengths
	
	# Get the list of fasta files that contained sequences of different lengths
	with open(f'{time.strftime('%Y-%m-%d')}_fasta_files_with_different_lengths.txt', 'a') as f:
		if len(len_set) > 1:
			f.write(f'{fasta_file}\n')
			print(f'Error: {fasta_file} contains sequences of different lengths.\
				{len_set}')
			return None

		else:
			# SeqIO.write(records, fasta_file + '.phylip', 'phylip-sequential') # Not the correct format for PAML, used ALTER instead
			SeqIO.write(records, fasta_file + '.phylip', 'phylip') # also not correct
			return fasta_file + '.phylip'


# def run_yn00(fasta_file): # this code works well, but you must use ALTER to convert the fasta file to phylip, not fasta2phylip()
# 	''' Run PAML yn00 method to calculate dN/dS ratio.
	
# 	Parameters:
# 		fasta_file (str): path to input FASTA file
# 	'''
	
# 	# Convert FASTA file to PHYLIP file
# 	inp = fasta2phylip(fasta_file) # input to PAML yn00 method
# 	inp_file = inp.split('/')[-1] # extract the file name
	
# 	# Run yn00 method
# 	if inp is not None:
# 		yn = yn00.Yn00(alignment=inp, working_dir=out_dir,
# 				 out_file = f"{out_dir}/{inp_file.replace('.fasta.phylip', '.yn00')}")
# 		yn.set_options(verbose = True,
# 			icode = 0,
# 			weighting = 0,
# 			commonf3x4 = 0,
# 			ndata = 1)
# 		results = yn.run(verbose=True)

# 		return results


if __name__ == '__main__':    
	# Parse command line arguments
	args = parse_args()
	cds_dir = args.cds_dir
	out_dir = args.out_dir

	# Get list of FASTA files
	os.chdir(cds_dir)
	cds_files = [f'{cds_dir}/{f}' for f in os.listdir('.') if f.endswith('_cds_aligned.fasta')]

	# Convert FASTA files to PHYLIP files
	pool = mp.Pool(mp.cpu_count()) # use all available cores
	pool.map(fasta2phylip, cds_files)
	pool.close()
