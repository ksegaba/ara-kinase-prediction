#!/usr/bin/env Rscript

# DESCRIPTION: Correction of raw fitness data for single and double mutants
# using Brianna's linear model to estimate the marginalized means of total seed count

# R v 4.4.1
# install.packages('emmeans')
# install.packages('dplyr')
# install.packages('plyr')
# install.packages('readxl')
# install.packages('lme4')
# install.packages('lmerTest')
# install.packages('pbkrtest')
library(emmeans)
library(dplyr)
library(plyr)
library(readxl)
library(lme4)

# Load the data
path <- 'data/20240917_melissa_ara_data/fitness_data_for_Kenia_09172024.xlsx'
meta <- read_xlsx(path, sheet='gene_names') # metadata
raw <- read_xlsx(path, sheet='with_border_cells') # raw data
dim(raw) # [1] 26886    28

# Remove sets that need to be excluded
exclude <- na.omit(meta[meta['Excluded by Kenia'] == 'y', 'Line'])
raw$Set <- as.character(raw$Set)
raw <- raw %>% filter(!Set %in% exclude$Line)
dim(raw) # [1] 25361    28

labels <- c('GN', 'PG', 'DTB', 'LN', 'DTF', 'SN', 'WO', 'FN', 'SPF', 'TSC', 'SH')
counter <- 1
for (label in labels){
    print(paste0('LABEL ', label, ' -----------------------------------------'))

    # If the label column is not numeric
    if (!is.numeric(raw[,label])) {
        raw[,label] <- as.numeric(raw[[label]])
    }

    # Collect fitted values for each label
    if (counter == 1) res <- data.frame()
    if (counter != 1) res2 <- data.frame()

    # Run linear regression per set
    for (set in unique(raw$Set)) {
        print(paste0('Set ', set))
        set_df <- raw[raw$Set == set, c('Set', 'Flat', 'Column', 'Row',
            'Number', 'Type', 'Genotype', 'Subline', label)] # set data

        # Remove rows with missing values in label
        set_df <- set_df[!is.na(set_df[,label]),]

        # For sets grown on multiple flats, include flat as random effect
        if (length(unique(set_df$Flat)) > 1) {
            # Set WT as the reference level
            set_df$Genotype <- factor(set_df$Genotype, levels=c('WT', 'MA', 'MB', 'DM'))

            # Fit a linear model
            model <- lmer(paste0(label, ' ~ Genotype + (1|Type) + (1|Flat)'), data = set_df)

            # Save the model
            saveRDS(model, paste0('data/20240917_melissa_ara_data/corrected_data/models/Set_', set,
                '_', label, '_linear_model.rds'))
            
            # Remove random effects from trait
            rand.eff <-ranef(model)
            set_df['tmp_label'] <- set_df[,label]
            
            for (f in unique(set_df$Flat)){ # remove flat effects
                flat_eff <- rand.eff$Flat[f,] * set_df[set_df$Flat == f, 'tmp_label']
                set_df[set_df$Flat == f, 'tmp_label'] <- set_df[set_df$Flat == f, 'tmp_label'] - flat_eff
            }
            for (t in unique(set_df$Type)){ # remove type effects
                type_eff <- rand.eff$Type[t,] * set_df[set_df$Type == t, 'tmp_label']
                set_df[set_df$Type == t, 'tmp_label'] <- set_df[set_df$Type == t, 'tmp_label'] - type_eff
            }
            set_df <- rename(set_df, c('tmp_label' = paste0(label, '_corrected')))

            # Calculate estimated marginal mean of the trait for each genotype
            emm <- as.data.frame(emmeans(model, ~ Genotype))
            emm <- rename_with(emm, ~ paste0(label, "_", .))
            out <- left_join(set_df, emm, by=c('Genotype'=paste0(label, '_Genotype')))
            if (counter == 1) res <- rbind.fill(res, out)
            if (counter != 1) res2 <- rbind.fill(res2, out)
        }
        if (length(unique(set_df$Flat)) == 1) {
            # Set WT as the reference level
            set_df$Genotype <- factor(set_df$Genotype, levels=c('WT', 'MA', 'MB', 'DM'))
            tryCatch({
                # Fit a linear model
                model <- lmer(paste0(label, ' ~ Genotype + (1|Type)'), data = set_df)

                # Save the model
                saveRDS(model, paste0('data/20240917_melissa_ara_data/corrected_data/models/Set_',
                    set, '_', label, '_linear_model.rds'))

                # Remove random effects from trait
                rand.eff <-ranef(model)
                set_df['tmp_label'] <- set_df[,label]

                for (t in unique(set_df$Type)){ # remove type effects
                    type_eff <- rand.eff$Type[t,] * set_df[set_df$Type == t, 'tmp_label']
                    set_df[set_df$Type == t, 'tmp_label'] <- set_df[set_df$Type == t, 'tmp_label'] - type_eff
                }
                set_df <- rename(set_df, c('tmp_label' = paste0(label, '_corrected')))

                # Calculate estimated means
                emm <- as.data.frame(emmeans(model, ~ Genotype))
                emm <- rename_with(emm, ~ paste0(label, "_", .))
                out <- left_join(set_df, emm, by=c('Genotype'=paste0(label, '_Genotype')))
                if (counter == 1) res <- rbind.fill(res, out)
                if (counter != 1) res2 <- rbind.fill(res2, out)
            }, error = function(e) {
                print(e)
            }, warning = function(w) {
                print(w)
            })
        }
    }

    if (counter != 1) {
        print(dim(res))
        print(dim(res2))
        res <- left_join(res, res2, by=c('Set', 'Flat', 'Column', 'Row', 
            'Number', 'Type', 'Genotype', 'Subline'), keep=F)
    }
    counter <- counter + 1
}

# Save the corrected data
write.table(res, paste0('data/20240917_melissa_ara_data/corrected_data/fitness_data_for_Kenia_09172024_corrected.tsv'), row.names=F, quote=F, sep='\t')
