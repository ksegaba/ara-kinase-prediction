Arabidopsis kinase genetic interaction prediction project

# Raw data:
* All the raw datasets used for this project are in the folder /home/seguraab/ara-kinase-prediction/data/

* __Paper:__ https://doi.org/10.1093/molbev/msab111 (Cusack et al., 2021 MBE)  

* __Data retrieval sources:__
    * __GitHub:__ https://github.com/ShiuLab/Manuscript_Code/tree/master/2021_Arabidopsis_redundancy_modeling  
    * __Zenodo:__ https://zenodo.org/records/3987384  
        * *Dataset_2.txt:* training set feature matrix (inclusive genetic redundancy gene pairs [RD9]) in /home/seguraab/ara-kinase-prediction/data/2021_cusack_supplementary_data/  
        * *Dataset_4.txt:* training set feature matrix (extreme genetic redundancy gene pairs [RD4]) and testing set (kinases) in /home/seguraab/ara-kinase-prediction/data/2021_cusack_supplementary_data/  
    * __From the MSU hpcc:__ /mnt/research/ShiuLab/21_arabidopsis_redundancy

* __Feature name list:__ 
    * *Supplemental_tables_revision.xlsx:* Supplemental table 10 (200 features) in /home/seguraab/ara-kinase-prediction/data/2021_cusack_supplementary_data/

* __Instances:__  
    * Training data: 
        * *interactions_fitness.txt:* Melissa's duplicate gene pair instances with binary genetic interaction label (0 for not interacting, 1 for interacting)
    * Test data: 
        * *FILE TBD:* 1000 kinase genes from *Dataset_4.txt* that are not overlapping with *interactions_fitness.txt*




