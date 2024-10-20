# Description: Calculate epistasis values for each plant set and fitness trait

library(readxl)
library(tidyr)
library(plyr)
library(dplyr)
library(lmerTest)
library(performance)

# Prepare fitness data
data <- read_excel(
  'data/20240923_melissa_ara_data/fitness_data_for_Kenia_09232024.xlsx',
  sheet='with_border_cells')

data$MA <- 0
data[data$Genotype == 'MA', 'MA'] <- 1
data$MB <- 0
data[data$Genotype == 'MB', 'MB'] <- 1
data$DM <- 0
data[data$Genotype == 'DM', 'DM'] <- 1
data[data$Genotype == 'DM', 'MA'] <- 1
data[data$Genotype == 'DM', 'MB'] <- 1

pseudolog10 <- function(x){
  # Pseudo-Logarithm of base 10, is defined for all real numbers. Use instead of
  # log10, which returns infinite/NaN values for x <= 0
  return ( log((x/2) + sqrt((x/2)^2 + 1)) )/log(10)
}

correct_batch <- function(set_df, label, formula){
  # Correct for batch effects due to flat number (Flat) and location on the flat (Type)

  model <- lmer(formula, data=set_df)

  # Remove random effects from trait
  rand.eff <- ranef(model)
  set_df['tmp_label'] <- set_df[,label]

  if (length(unique(set_df$Flat)) > 1) {
    for (f in unique(set_df$Flat)){ # remove flat effects
      flat_eff <- rand.eff$Flat[f,] * set_df[set_df$Flat == f, 'tmp_label']
      set_df[set_df$Flat == f, 'tmp_label'] <- set_df[set_df$Flat == f, 'tmp_label'] - flat_eff
    }
  }
  
  for (t in unique(set_df$Type)){ # remove type effects
      type_eff <- rand.eff$Type[t,] * set_df[set_df$Type == t, 'tmp_label']
      set_df[set_df$Type == t, 'tmp_label'] <- set_df[set_df$Type == t, 'tmp_label'] - type_eff
  }
  set_df <- plyr::rename(set_df, c('tmp_label' = paste0(label, '_corrected')))
  
  return(list(set_df=set_df, model=model))
}

get_epi_stats <- function(set, label, model, formula='', set_df=''){
  # Calculate the epistasis value
  # print(model)
  result <- tryCatch({
    if (formula != '') {
      model <- lm(formula, data=set_df) # don't leave set_df = '', set it as set_df = set_df
      e_est <- as.numeric(coef(model)['MA:MB'])
      lowerCI <- confint(model)['MA:MB',1]
      upperCI <- confint(model)['MA:MB',2]
      rsquared <- summary(model)$adj.r.squared
      pval_e <- as.numeric(summary(model)$coefficients[,4]['MA:MB'])
    } else {
      # Extract stats of interaction term
      e_est <- as.numeric(fixef(model)['MA:MB'])
      lowerCI <- confint(model, level=0.95)['MA:MB',1]
      upperCI <- confint(model, level=0.95)['MA:MB',2]
      r2_model <- r2_nakagawa(model, tolerance=1e-1000)
      rsquared <- as.numeric(r2_model[[2]]) # Marginal R-squared takes into account only the variance of the fixed effects
      pval_e <- as.numeric(coef(summary(model))[,'Pr(>|t|)']['MA:MB'])
    }

    # Return stats
    return(c(set, e_est, lowerCI, upperCI, rsquared, pval_e, label,  'MA:MB'))
  }, error = function(e) {
    warning("An error occurred:", conditionMessage(e))
    return(c(set, rep(NA, 5), label, 'MA:MB')) # Return a vector of names
  })

  return(result)
}

get_epi_stats2 <- function(set, label, model, formula='', set_df=''){
  # Calculate the epistasis value
  
  result <- tryCatch({
    if (formula != ''){
      # Use if the label is already batch corrected
      model <- lm(formula, data=set_df) # don't leave set_df = '', set it as set_df = set_df
      e_est <- as.numeric(coef(model)['DM'])
      lowerCI <- confint(model)['DM',1]
      upperCI <- confint(model)['DM',2]
      rsquared <- summary(model)$adj.r.squared
      pval_e <- as.numeric(summary(model)$coefficients[,4]['DM'])

    } else {
      # Extract stats of interaction term
      e_est <- as.numeric(fixef(model)['DM'])
      lowerCI <- confint(model, level=0.95)['DM',1]
      upperCI <- confint(model, level=0.95)['DM',2]
      r2_model <- r2_nakagawa(model, tolerance=1e-1000)
      rsquared <- as.numeric(r2_model[[2]]) # Marginal R-squared takes into account only the variance of the fixed effects
      pval_e <- as.numeric(coef(summary(model))[,'Pr(>|t|)']['DM'])
    }
    
    # Calculate the estimated marginal means relative to wild type
    emm <- emmeans(model, '~ MA + MB + MA:MB')

    # Return stats
    return(c(set, e_est, lowerCI, upperCI, rsquared, pval_e, label, 'DM'))
  }, error = function(e) {
    warning("An error occurred:", conditionMessage(e))
    return(c(set, rep(NA, 5), label, 'DM')) # Return a vector of names
  })

  return(result)
}

# Calculate the epistasis values for each plant set and fitness label
labels <- c('PG', 'DTB', 'LN', 'DTF', 'SN', 'WO', 'FN', 'SPF', 'TSC', 'SH')
counter <- 1
for (label in labels){
  print(paste0('LABEL ', label, ' -----------------------------------------'))

  # If the label column is not numeric
  if (!is.numeric(data[,label])) {
      data[,label] <- as.numeric(data[[label]])
  }

  # Collect epistasis values for each label and the batch corrected data
  if (counter == 1) {
    res <- data.frame()
    batch_corrected_df <- data.frame()
    # batch_corrected_df2 <- data.frame()
  }

  if (counter != 1) {
    res2 <- data.frame()
    batch_corrected_dfb <- data.frame()
    # batch_corrected_df2b <- data.frame()
}

  # Run linear regression per set
  for (set in unique(data$Set)) {
    # print(paste0('Set ', set))
    set_df <- data[data$Set == set, c("Set", "Flat", "Column", "Row",
                  "Number", "Type", "Genotype", "Subline", "MA", "MB", "DM",
                  label)] # set data

    # Remove rows with missing values in label
    set_df <- set_df[!is.na(set_df[,label]),]
    if (nrow(set_df) == 0) next

    # Batch correction
    pop_mean <- colMeans(set_df[set_df$Genotype == 'WT', label])
    set_df$pop_mean <- as.numeric(pop_mean) # population mean
    
    if (length(unique(set_df$Flat)) > 1) {
      correction_res <- correct_batch(set_df, label, paste0(label, ' ~ MA + MB + MA:MB + pop_mean + (1|Type) + (1|Flat)'))
      corrected_df <- correction_res$set_df
      model <- correction_res$model

      # correction_res2 <- correct_batch(set_df, label, paste0(label, ' ~ MA + MB + DM + pop_mean + (1|Type) + (1|Flat)'))
      # corrected_df2 <- correction_res2$set_df
      # model2 <- correction_res2$model
    } else {
      correction_res <- correct_batch(set_df, label, paste0(label, ' ~ MA + MB + MA:MB + pop_mean + (1|Type)'))
      corrected_df <- correction_res$set_df
      model <- correction_res$model

      # correction_res2 <- correct_batch(set_df, label, paste0(label, ' ~ MA + MB + DM + pop_mean + (1|Type)'))
      # corrected_df2 <- correction_res2$set_df
      # model2 <- correction_res2$model
    }

    # Apply transformations to the ORIGINAL label --- Fix 10/15/2024
    ## pseudolog10
    corrected_df[, paste0(label, '_corrected_plog10')] <- pseudolog10(corrected_df[, paste0(label, '_corrected')])
    # corrected_df2[, paste0(label, '_corrected_plog10')] <- pseudolog10(corrected_df2[, paste0(label, '_corrected')])

    ## add 1 to count traits
    if (label %in% list('SN', 'WO', 'FN', 'TSC')){
      corrected_df[, paste0(label, '_corrected_plus1')] <- corrected_df[, paste0(label, '_corrected')] + 1
      # corrected_df2[, paste0(label, '_corrected_plus1')] <- corrected_df2[, paste0(label, '_corrected')] + 1
      
      ## calculate the log10
      corrected_df[,paste0(label, '_corrected_plus1_log10')] <- log10(corrected_df[, paste0(label, '_corrected_plus1')])
      # corrected_df2[,paste0(label, '_corrected_plus1_log10')] <- log10(corrected_df2[, paste0(label, '_corrected_plus1')])
    } else {
      # calculate the log10
      corrected_df[,paste0(label, '_corrected_log10')] <- log10(corrected_df[, paste0(label, '_corrected')])
      # corrected_df2[,paste0(label, '_corrected_log10')] <- log10(corrected_df2[, paste0(label, '_corrected')])
    }

    # Calculate the epistasis value
    for (new_label in c(paste0(label, '_corrected'),# paste0(label, '_corrected_log10'),
      paste0(label, '_corrected_plog10'))){#, # paste0(label, '_corrected_plus1'), 
      #paste0(label, '_corrected_plus1_log10'))){
      
      if (new_label %in% colnames(corrected_df)){
        if (endsWith(new_label, '_corrected')) {
          out <- get_epi_stats(set, new_label, model=model) # has MA:MB interaction term
          # out2 <- get_epi_stats2(set, new_label, model=model2) # has DM term inplace of MA:MB
        } else {
          # if (endsWith(new_label, '_log10')) {
          #   corrected_df_sub <- corrected_df[!is.infinite(corrected_df[[new_label]]),]
            # corrected_df2_sub <- corrected_df2[!is.infinite(corrected_df2[[new_label]]),]
            # if (nrow(corrected_df_sub) == 0 | nrow(corrected_df2_sub) == 0) next
            # out <- get_epi_stats(set, new_label, '', formula=paste0(new_label, ' ~ MA + MB + MA:MB'), set_df=corrected_df_sub)
            # out2 <- get_epi_stats2(set, new_label,  '', formula=paste0(new_label, ' ~ MA + MB + DM'), set_df=corrected_df2_sub)
          # }
          out <- get_epi_stats(set, new_label, '', formula=paste0(new_label, ' ~ MA + MB + MA:MB'), set_df=corrected_df)
          # out2 <- get_epi_stats2(set, new_label,  '', formula=paste0(new_label, ' ~ MA + MB + DM'), set_df=corrected_df2)
        }
      
        # Collect the results
        if (counter == 1) {
          res <- rbind(res, out) #, out2)
          batch_corrected_df <- rbind.fill(batch_corrected_df, corrected_df)
          # batch_corrected_df2 <- rbind.fill(batch_corrected_df2, corrected_df2)
        }

        if (counter != 1) {
          res2 <- rbind(res2, out) #, out2)
          batch_corrected_dfb <- rbind.fill(batch_corrected_dfb, corrected_df)
          # batch_corrected_df2b <- rbind.fill(batch_corrected_df2b, corrected_df2)
        }
      } # ensure the transformed label is in the batch corrected data
    } # all transformed labels are done
  } # all sets are done

  if (counter == 1) {
    colnames(res) <- c('Set', 'e_est', 'lowerCI', 'upperCI', 'rsquared', 'pval_e', 'Label', 'Term')
  }

  if (counter != 1) {
    # epistasis data
    print(dim(res))
    print(dim(res2))
    colnames(res2) <- c('Set', 'e_est', 'lowerCI', 'upperCI', 'rsquared', 'pval_e', 'Label', 'Term')
    res <- rbind.fill(res, res2)

    # batch corrected data
    batch_corrected_df <- left_join(batch_corrected_df, batch_corrected_dfb,
      by=c('Set', 'Flat', 'Column', 'Row', 'Number', 'Type', 'Genotype', 'Subline'), keep=F)
    # batch_corrected_df2 <- left_join(batch_corrected_df2, batch_corrected_df2b,
    #   by=c('Set', 'Flat', 'Column', 'Row', 'Number', 'Type', 'Genotype', 'Subline'), keep=F)
  }
  counter <- counter + 1
}

# Save the batch corrected data
write.table(batch_corrected_df,
  paste0('data/20240923_melissa_ara_data/corrected_data/fitness_data_for_Kenia_09232024_corrected_withInterTerm.tsv'),
  row.names=F, quote=F, sep='\t')
# write.table(batch_corrected_df2,
#   paste0('data/20240923_melissa_ara_data/corrected_data/fitness_data_for_Kenia_09232024_corrected_withDMTerm.tsv'),
#   row.names=F, quote=F, sep='\t')

# Determine direction of epistasis
df_results <- res %>% mutate(
  Epistasis_Direction = case_when(lowerCI < 0 & upperCI < 0  ~ 'Negative',
                                  lowerCI > 0 & upperCI > 0 ~ 'Positive',
                                  lowerCI <=  0 & upperCI >= 0 ~ 'Not Detected'))
table(df_results$Epistasis_Direction)
table(df_results$Label)

# Save the epistasis results
write.csv(df_results,
  paste0('data/20240923_melissa_ara_data/corrected_data/fitness_data_for_Kenia_09232024_corrected_epistasis_linear.csv'),
  row.names=F, quote=F)
