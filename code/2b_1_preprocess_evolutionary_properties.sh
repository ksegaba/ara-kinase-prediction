#!/bin/bash
:'
Pre-process evolutionary properties feature data as described by Cusack et al., 2021. 

First: Query Arabidopsis thaliana protein sequences with BLASTp against 4 species
  - Theobroma cacao
  - Populus trichocarpa
  - Glycine max
  - Solanum lycopersicum

Second: 
  - Generate protein sequence alignments with MUSCLE from BLASTp results
  - Generate nucleotide coding sequence alignments with MUSCLE

Third: Build gene trees from the coding sequence alignments with RAxML

Fourth:
  - Calculate the evolutionary rate (Ka/Ks) for each gene with PAML
  - Calculate the nucleotide and amino acid sequence similarity with EMBOSS Needle
'
# Author: Kenia Segura Abá

######################## First. Run the BLASTp searches ########################
run_blastp() {
  # Description: Run BLASTp and reciprocal BLASTp searches
  
  # Arguments for the BLAST database
  local db_path="$1"      # Path to save subject species BLAST database
  local db_fasta="$2"     # Protein FASTA file for the subject species
  local out_db="$3"       # File name of the BLAST database

  # Arguments for the BLASTp searches
  local save_path="$4"    # Path to save the BLASTp results
  local save_prefix="$5"  # Prefix for the BLASTp results file
  
  # Create a BLAST database of protein sequences for the subject species
  makeblastdb -in ${db_path}/${db_fasta} -out ${db_path}/${out_db} -dbtype prot

  # Run BLASTp
  diamond prepdb -d ${db_path}/${out_db}
  diamond blastp --threads 4 --query ${At_p}/Athaliana_167_TAIR10.protein.fa \
    --db ${db_path}/${out_db} \
    --max-target-seqs 1 --max-hsps 1 --evalue 1e-6 --no-parse-seqids \
    --outfmt 6 qseqid stitle qlen slen qstart qend sstart send length evalue bitscore pident \
    --out ${save_path}/${save_prefix}_blastp.txt

  # Run reciprocal BLASTp
  diamond prepdb -d ${At_p}/Athaliana_db
  diamond blastp --threads 4 --query ${db_path}/${db_fasta} \
    --db ${At_p}/Athaliana_db \
    --max-target-seqs 1 --max-hsps 1 --evalue 1e-6 --no-parse-seqids \
    --outfmt 6 qseqid stitle qlen slen qstart qend sstart send length evalue bitscore pident \
    --out ${save_path}/${save_prefix}_reciprocal_blastp.txt
}

# Paths to the protein FASTA files for each species
At_p=/home/seguraab/ara-kinase-prediction/data/TAIR10
Tc_p=/home/seguraab/ara-kinase-prediction/data/NCBI_genomes/GCF_000208745.1
Gm_p=/home/seguraab/ara-kinase-prediction/data/NCBI_genomes/GCF_000004515.6
Sl_p=/home/seguraab/ara-kinase-prediction/data/NCBI_genomes/GCF_000188115.5
Pt_p=/home/seguraab/ara-kinase-prediction/data/NCBI_genomes/GCF_000002775.5

# Create a BLAST database of protein sequences for A. thaliana (for the reciprocal BLASTp)
makeblastdb -in ${At_p}/Athaliana_167_TAIR10.protein.fa \
  -out ${At_p}/Athaliana_db -dbtype prot -title "A. Thaliana Protein Sequences"

# Run the BLASTp and the reciprocal BLASTp searches for each subject species
out_path=/home/seguraab/ara-kinase-prediction/data/2021_cusack_data/evolutionary_properties_data/0_blast_res
run_blastp $Tc_p "protein.faa" "Tcacao_db" $out_path "TAIR10_Tcacao"
run_blastp $Gm_p "protein.faa" "Gmax_db" $out_path "TAIR10_Gmax"
run_blastp $Sl_p "protein.faa" "Slycopersicum_db" $out_path "TAIR10_Slycopersicum"
run_blastp $Pt_p "protein.faa" "Ptrichocarpa_db" $out_path "TAIR10_Ptrichocarpa"

################ Second. Protein sequence Alignments with MUSCLE ###############
convert_to_one_liner() {
  # Description: Convert multi-line sequences in a FASTA file into one-liners

  local fasta_file="$1"  # Protein FASTA file
  awk '/^>/ {printf("\n%s\n",$0); next;} {printf("%s",$0);} END {printf("\n");}' < $fasta_file >> $fasta_file'_one_liner.fa'
}


align_seqs() {
  :'Align pairs of protein sequences with MUSCLE given a tabular blastp results file.
  1. Create the gene set (query and subject genes) FASTA file
  2. Align the gene set with MUSCLE
  '
  
  # Arguments
  local gene_file="$1"    # Tabular file with BLASTp gene sets (per line: Arabidopsis_gene subject_gene_1 subject_gene_2 subject_gene_3 subject_gene_4)
  local At_fasta="$2"     # One-liner protein FASTA file for Arabidopsis thaliana
  local s_fasta1="$3"     # One-liner protein FASTA file for the subject species gene 1
  local s_fasta2="$4"     # One-liner protein FASTA file for the subject species gene 2
  local s_fasta3="$5"     # One-liner protein FASTA file for the subject species gene 3
  local s_fasta4="$6"     # One-liner protein FASTA file for the subject species gene 4
  local save_prefix="$7"  # Prefix for the alignment output files

  # Determine the number of reciprocal best hits (This is irrelevant, I was just curious)
  # $blastp_file was the prefix argument for the BLASTp results files
  # awk -F' ' '{if ($1 < $2) print $1, $2; else print $2, $1}' $blastp_file'_blastp.txt' | sort -o $blastp_file'_blastp_sorted_pairs.txt'  
  # awk -F' ' '{if ($1 < $2) print $1, $2; else print $2, $1}' $blastp_file'_reciprocal_blastp.txt' | sort -o $blastp_file'_reciprocal_blastp_sorted_pairs.txt'
  # comm -12 $blastp_file'_blastp_sorted_pairs.txt' $blastp_file'_reciprocal_blastp_sorted_pairs.txt' >> $blastp_file'_shared_blastp_hits.txt'
  # cat $blastp_file'_shared_blastp_hits.txt' | wc -l

  ### Create the gene pair FASTA file and align with MUSCLE
  # Progress bar
  total=$(cat $gene_file | wc -l)
  progress=0
  
  while IFS= read -r line
  do
    ## Write the gene pair protein sequences to a new FASTA file
    read -r at_gene s_gene1 s_gene2 s_gene3 s_gene4 <<< $line # gene set

    # Extract the A. thaliana sequences from the FASTA file
    grep -A 1 $at_gene $At_fasta >> ${save_prefix}_gene_pair_protein.fasta
    
    # Extract the subject species sequences from the FASTA file
    if [[ $s_gene1 != "NA" ]]; then
      grep -A 1 $s_gene1 $s_fasta1 >> ${save_prefix}_gene_pair_protein.fasta
    fi
    if [[ $s_gene2 != "NA" ]]; then
      grep -A 1 $s_gene2 $s_fasta2 >> ${save_prefix}_gene_pair_protein.fasta
    fi
    if [[ $s_gene3 != "NA" ]]; then
      grep -A 1 $s_gene3 $s_fasta3 >> ${save_prefix}_gene_pair_protein.fasta
    fi
    if [[ $s_gene4 != "NA" ]]; then
      grep -A 1 $s_gene4 $s_fasta4 >> ${save_prefix}_gene_pair_protein.fasta
    fi
  
    ## Align the protein sequences
    muscle -align $save_prefix'_gene_pair_protein.fasta' \
      -output ${save_prefix}_${at_gene}_alignment.fasta
    
    rm ${save_prefix}_gene_pair_protein.fasta # delete the gene pair FASTA file

    # Update progress bar
    progress=$((progress + 1))
    echo -ne "Progress: $progress/$total\r"
  
  done < ${gene_file}
}


# Paths to the protein FASTA files for each species
At_p=/home/seguraab/ara-kinase-prediction/data/TAIR10/Athaliana_167_TAIR10.protein.fa
Tc_p=/home/seguraab/ara-kinase-prediction/data/NCBI_genomes/GCF_000208745.1/protein.faa
Gm_p=/home/seguraab/ara-kinase-prediction/data/NCBI_genomes/GCF_000004515.6/protein.faa
Sl_p=/home/seguraab/ara-kinase-prediction/data/NCBI_genomes/GCF_000188115.5/protein.faa
Pt_p=/home/seguraab/ara-kinase-prediction/data/NCBI_genomes/GCF_000002775.5/protein.faa

# Convert the whole genome protein FASTA files into one-liners
convert_to_one_liner $At_p
convert_to_one_liner $Tc_p
convert_to_one_liner $Gm_p
convert_to_one_liner $Sl_p
convert_to_one_liner $Pt_p

# Combine the top gene hits from all species into one file
conda activate py310
python /home/seguraab/ara-kinase-prediction/code/2b_1_combine_blastp_res.py
conda deactivate

# Run the alignment for each Arabidopsis gene with its top hits in all the other species
data_path=/home/seguraab/ara-kinase-prediction/data/evolutionary_properties_data
align_seqs "${data_path}/0_blast_res/blastp_TAIR10_top_hits_all.txt" \
  $At_p"_one_liner.fa" \
  $Gm_p"_one_liner.fa" \
  $Tc_p"_one_liner.fa" \
  $Pt_p"_one_liner.fa" \
  $Sl_p"_one_liner.fa" \
  "${data_path}/1_muscle_res/blastp_TAIR10_top_hits_all"

################## Third. Gene Trees with RAxML #####################
# Back translate protein sequence alignments to nucleotide sequence Alignments


