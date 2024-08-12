#!/usr/bin/env python3
'''
Back translate a peptide sequence alignment to nucleotide coding sequence 
alignment.

The code is adapted from the original code written by Dr. Shin-Han Shiu 
(AlnUtility.py, Translation.py, and FastaManager.py)
'''

import argparse, os, re

def parse_args():
	'''
	Parse command line arguments.
	#
	Returns:
		- args (argparse.Namespace): parsed arguments
	'''
	parser = argparse.ArgumentParser(description='Back translate a peptide sequence alignment to nucleotide coding sequence alignment.')
	
	parser.add_argument('-dir', type=str, help='Path to peptide sequence alignment file folder.')
	
	args = parser.parse_args()
	
	return args


def get_nt_code():
	'''
	Function that returns a codon table that maps nucleotide codons to amino 
	acids.
	Adapted from the get_nt_code function in Translation.py written by 
	Dr. Shin-Han Shiu ().
	'''
	
	# /: between codon frameshift, reported by e.g. tfasty
	# |: within codon frameshift
	code = {"TTT":"F","TCT":"S","TAT":"Y","TGT":"C",
			"TTC":"F","TCC":"S","TAC":"Y","TGC":"C",
			"TTA":"L","TCA":"S","TAA":"*","TGA":"*",
			"TTG":"L","TCG":"S","TAG":"*","TGG":"W",
			
			"CTT":"L","CCT":"P","CAT":"H","CGT":"R",
			"CTC":"L","CCC":"P","CAC":"H","CGC":"R",
			"CTA":"L","CCA":"P","CAA":"Q","CGA":"R",
			"CTG":"L","CCG":"P","CAG":"Q","CGG":"R",
			
			"ATT":"I","ACT":"T","AAT":"N","AGT":"S",
			"ATC":"I","ACC":"T","AAC":"N","AGC":"S",
			"ATA":"I","ACA":"T","AAA":"K","AGA":"R",
			"ATG":"M","ACG":"T","AAG":"K","AGG":"R",
			
			"GTT":"V","GCT":"A","GAT":"D","GGT":"G",
			"GTC":"V","GCC":"A","GAC":"D","GGC":"G",
			"GTA":"V","GCA":"A","GAA":"E","GGA":"G",
			"GTG":"V","GCG":"A","GAG":"E","GGG":"G",
			
			"NNN":"X","---":"-","///":"Z","|||":"Z"}
	
	return code


def back_translate(pep, nt):
	'''
	Back translate a peptide sequence alignment to nucleotide. Adapted from 
	the back_translate2 function in Translation.py written by Dr. Shin-Han Shiu.
	#
	Parameters:
		- pep (str): peptide sequence alignment (contains '-')
		- nt (str): nucleotide coding sequence
	#
	Returns:
		- seq (str): nucleotide coding sequence alignment
	'''
	
	# remove line breaks
	pep = rmlb(pep)
	nt = rmlb(nt)
	
	# verify peptide length
	plen = 0
	if pep.find('-') != -1:
		plist = re.split('-+', pep)
		for i in plist:
			plen += len(i)
	else:
		plen = len(pep)
	
	# nt sequences don't have '-', but need to rid of stop
	if nt[-3:] in ["TAA","TGA","TAG","taa","tga","tag"]:
		nt = nt[:-3]
	
	# check if nt sequence is a multiple of 3
	flag = 0
	if len(nt) % 3 != 0:
		print('Error: nt sequence length is not a multiple of 3')
		print('Size discrepancy: nt', len(nt), 'pep', plen)
		flag = 1
	
	# align the nt sequence to the pep sequence alignment
	seq = ''
	if not flag: # continue if no error
		countN = 0
		codes = get_nt_code()
		for i in pep:
			if i == '-':
				seq += '---'
			else:
				seq += nt[countN:countN+3]
				
				# check if the translation is right
				if nt[countN:countN+3].upper() not in codes.keys():
					print('Error: unknown codon', i.upper(),\
						  nt[countN:countN+3].upper())
				elif codes[nt[countN:countN+3].upper()] != i.upper():
					print('Error: translation discrepancy', i.upper(),\
						  codes[nt[countN:countN+3].upper()])
				
				countN += 3
	
	return seq, flag


def rmlb(astr):
	'''
	Remove line breaks from a string.
	Adapted from the rmlb function in FastaManager.py written by Dr. Shin-Han 
	Shiu.
	
	Parameters:
		- astr (str): string with line breaks
	
	Returns:
		- astr (str): string without line breaks
	'''
	
	if astr[-2:] == "\r\n":
		astr = astr[:-2]
	elif astr[-1] == "\n":
		astr = astr[:-1]
	
	return astr


def fasta_to_dict(fasta, ref_seq='n'):
	'''
	Return a dictionary with sequence IDs as keys and sequences as values.
	Adapted from the fasta_to_dict function in FastaManager.py written by 
	Dr. Shin-Han Shiu.
	
	Parameters:
		- fasta (str): path to fasta file
	
	Returns:
		- fasta_dict (dict): dictionary with sequence IDs as keys and sequences 
							 as values
	'''
	
	# read the fasta file
	with open(fasta, 'r') as inp:
		inl = inp.readline()
		fasta_dict = {} # dictionary to store sequences
		c = 0 # sequence counter
		N = 0 # dict key counter (non-redundant IDs)
		while inl != '':
			inl = rmlb(inl) # remove line breaks
			next = 0
			idx = '' # sequence ID
			desc = '' # sequence description
			
			if inl == '':
				pass
			elif inl[0] == '>': # start of a new sequence entry
				print (f'{c % 1e3} k')
				c += 1
				
				if ref_seq == 'n': # non-NCBI RefSeq genomes
					# delimit the sequence ID and description
					if inl.find(' ') != -1:
						desc = inl[inl.find(' ')+1:] # sequence description
						idx = inl[1:inl.find(' ')] # sequence ID
					else:
						idx = inl[1:] # sequence ID
				
				else: # NCBI RefSeq genomes
					match = re.search(f'protein_id=(.*?)\]', inl)
					if match:
						idx = match.group(1) # protein sequence identifier
					else:
						idx = inl[1:]
				
				# count lines and store sequences into a list
				seq_list = []
				inl = inp.readline()
				while inl[0] != '>':
					inl = rmlb(inl) #.strip()
					# inl = inl + '\n' # add new line
					seq_list.append(inl)
					inl = inp.readline()
					if inl == '':
						break
				
				seq = ''.join(seq_list)
				
				# store sequence into dictionary
				if idx in fasta_dict.keys():
					print('Error: redundant sequence ID', idx)
					
					if len(fasta_dict[idx][1]) < len(seq):
						fasta_dict[idx] = [desc, seq]
						print('longer')
					else:
						print('shorter')
						
				else:
					N += 1
					fasta_dict[idx] = [desc, seq]
				
				next = 1
				
			if not next: # no extra line is read because of the inner while loop
				inl = inp.readline()
		
		print(f'Total {c} sequences, {N} with non-redundant IDs')
	
	return fasta_dict


def get_cds_align(pep_file, cds_out):
	'''
	Obtain the nucleotide coding sequence alignments for each peptide sequence 
	alignment within a file.
	#
	Parameters:
		- pep_file (str): path to peptide sequence alignment file
		- cds_out (str): path to output nucleotide coding sequence alignment file
	'''
	
	# read peptide alignment fasta into a dict
	pep_dict = fasta_to_dict(pep_file)
	
	with open(cds_out + '_cds_aligned.fasta', 'w') as oup:
		flag = 0
		for pep in pep_dict.keys():
			ref_seq = 'y'
			
			# determine which sequence belongs to which species
			if 'TAIR10' in pep_dict[pep][0]:
				cds_file = '/home/seguraab/ara-kinase-prediction/data/TAIR10/Athaliana_167_TAIR10.cds.fa'
				ref_seq = 'n'
			
			if 'Theobroma cacao' in pep_dict[pep][0]:
				cds_file = '/home/seguraab/ara-kinase-prediction/data/NCBI_genomes/GCF_000208745.1/cds_from_genomic.fna'
			
			if 'Glycine max' in pep_dict[pep][0]:
				cds_file = '/home/seguraab/ara-kinase-prediction/data/NCBI_genomes/GCF_000004515.6/cds_from_genomic.fna'
			
			if 'Populus trichocarpa' in pep_dict[pep][0]:
				cds_file = '/home/seguraab/ara-kinase-prediction/data/NCBI_genomes/GCF_000002775.5/cds_from_genomic.fna'
			
			if 'Solanum lycopersicum' in pep_dict[pep][0]:
				cds_file = '/home/seguraab/ara-kinase-prediction/data/NCBI_genomes/GCF_000188115.5/cds_from_genomic.fna'
			
			# read species coding sequence fasta into a dict
			cds_dict = fasta_to_dict(cds_file, ref_seq)
			
			# back translate the peptide squences to nucleotide coding sequences
			seq, flag = back_translate(pep_dict[pep][1], cds_dict[pep][1])
			if flag:
				print('Error:', pep)
			
			flag = 0 # reset flag
			
			oup.write(f'>{pep}\n{seq}\n')


if __name__ == '__main__':

	args = parse_args() # Argument parsing

	muscle_res = args.dir # MUSCLE results directory
	pep_files = [file for file in os.listdir(muscle_res)] # protein sequence alignment files
	
	os.chdir(muscle_res) # Save CDS alignment files in the same directory
	if not os.path.exists('cds_alignments'):
		os.mkdir('cds_alignments') # Create a directory to save CDS alignment files
	
	os.chdir('cds_alignments') # Change to the new directory

	# Align peptide sequence alignment to nucleotide coding sequences
	for file in pep_files:
		cds_out = file.split('_alignment.fasta')[0]
		get_cds_align('../' + file, cds_out)