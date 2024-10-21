# Description: Calculate epistasis values for each plant set and fitness trait

library(readxl)
library(tidyr)
library(plyr)
library(dplyr)
library(lmerTest)
library(performance)
library(emmeans)

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
  # Note: Sometimes removing the random effects results in negative numbers.
  # Thus, instead of removing the batch effects, I will just take into account
  # the random effects in the linear model but I won't remove their effects to
  # correct the trait. This function has no purpose now. Leaving for record-keeping.
  
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

get_epi_stats <- function(set, label, set_df, formula){
  # Calculate the epistasis values and genotype estimated marginal mean of
  # the label of interest with a linear model
  
  result <- tryCatch({
    model <- lm(formula, data=set_df)
    # model <- lmer(formula, data=set_df)

    # Extract stats of interaction term
    e_est <- as.numeric(coef(model)['MA:MB'])
    lowerCI <- confint(model)['MA:MB',1]
    upperCI <- confint(model)['MA:MB',2]
    rsquared <- summary(model)$adj.r.squared
    pval_e <- as.numeric(summary(model)$coefficients[,4]['MA:MB'])

    # Calculate genotype estimated marginal means
    emm <- as.data.frame(emmeans(model, ~ MA + MB + MA:MB))
    emm$Genotype <- ''
    for (i in 1:nrow(emm)){
      if (emm$MA[i] == 1 & emm$MB[i] == 1) {
        emm$Genotype[i] <- 'DM'
      } else if (emm$MA[i] == 1 & emm$MB[i] == 0) {
        emm$Genotype[i] <- 'MA'
      } else if (emm$MA[i] == 0 & emm$MB[i] == 1) {
        emm$Genotype[i] <- 'MB'
      } else if (emm$MA[i] == 0 & emm$MB[i] == 0) {
        emm$Genotype[i] <- 'WT'
      }
    }
    emm <- rename_with(emm, ~ paste0(label, "_", .))
    out_df <- left_join(set_df, emm, by=c('Genotype'=paste0(label, '_Genotype')))
    out_df <- out_df[, !(colnames(out_df) %in% c(paste0(label, '_MA'), paste0(label, '_MB')))]
  
    # Return stats
    return(list(
      results=c(set, e_est, lowerCI, upperCI, rsquared, pval_e, label, 'MA:MB'),
      out_df=out_df))
  }, error = function(e) {
    warning("An error occurred:", conditionMessage(e))
    return(list(results=c(set, rep(NA, 5), label, 'MA:MB'), out_df=set_df)) # Return a vector of names
  })

  return(result)
}

# Calculate the epistasis values for each plant set and fitness label
labels <- c('PG', 'DTB', 'LN', 'DTF', 'SN', 'WO', 'FN', 'SPF', 'TSC', 'SH')
counter <- 1
for (label in labels){
  print(paste0('LABEL ', label, ' -----------------------------------------'))
  
  join_cols <- c('Set', 'Flat', 'Column', 'Row', 'Number', 'Type', 'Genotype',
    'Subline', 'MA', 'MB', 'DM', label)

  # If the label column is not numeric
  if (!is.numeric(data[,label])) data[,label] <- as.numeric(data[[label]])
  
  # Collect the epistasis values and genotype estimated marginal means for a `label`
  if (counter == 1) res <- data.frame() # collect epistasis values of the first label
  if (counter != 1) res2 <- data.frame() # for the other labels

  # Apply transformations to the ORIGINAL label
  ## pseudolog10
  data[, paste0(label, '_plog10')] <- pseudolog10(data[, label])
  
  ## add 1 to count traits
  if (label %in% list('SN', 'WO', 'FN', 'TSC')){
    data[, paste0(label, '_plus1')] <- data[, label] + 1
    
    ## calculate the log10
    data[, paste0(label, '_plus1_log10')] <- log10(data[, paste0(label, '_plus1')])
  } else {
    ## calculate the log10
    data[, paste0(label, '_log10')] <- log10(data[, label])
  }
  
  counter2 <- 1
  for (new_label in c(label, paste0(label, '_log10'), paste0(label, '_plus1'),
    paste0(label, '_plog10'), paste0(label, '_plus1_log10'))){

    # Collect the genotype estimated marginal means for a `label`
    if (counter2 == 1) emm_df <- data.frame() # always original `label`
    if (counter2 != 1) emm_df2 <- data.frame() # the other transformed labels

    if (new_label %in% colnames(data)){

      # Run linear regression per set
      progress <- 1
      for (set in unique(data$Set)) {

        if (label %in% list('SN', 'WO', 'FN', 'TSC')) {
          set_df <- data[data$Set == set, c("Set", "Flat", "Column", "Row",
            "Number", "Type", "Genotype", "Subline", "MA", "MB", "DM",
            label, paste0(label, '_plus1'), paste0(label, '_plus1_log10'),
            paste0(label, '_plog10'))] # set data
        } else {
            set_df <- data[data$Set == set, c("Set", "Flat", "Column", "Row",
              "Number", "Type", "Genotype", "Subline", "MA", "MB", "DM",
              label, paste0(label, '_log10'), paste0(label, '_plog10'))]
        }

        set_df <- set_df[!is.na(set_df[,label]),] # Remove rows with missing values in label
        if (nrow(set_df) == 0) next # skip sets with no data
        set_df$Type <- factor(set_df$Type) # set Type as a factor

        # Calculate the population mean
        pop_mean <- colMeans(set_df[set_df$Genotype == 'WT', new_label])
        set_df$pop_mean <- as.numeric(pop_mean) # population mean
      
        # Define the linear equation
        if (length(unique(set_df$Flat)) > 1) {
          set_df$Flat <- factor(set_df$Flat)
          formula <- paste0(new_label, ' ~ MA + MB + MA:MB + pop_mean + Type + Flat')
        } else {
          formula <- paste0(new_label, ' ~ MA + MB + MA:MB + pop_mean + Type')
        }

        # Calculate the epistasis statistics
        if (endsWith(new_label, '_log10')) { # deal with infinite values
          set_df_sub <- set_df[!is.infinite(set_df[[new_label]]),]
          if (nrow(set_df_sub) == 0) {
            next
          } else {
            out <- get_epi_stats(set, new_label, set_df_sub, formula)
          }
        } else {
          out <- get_epi_stats(set, new_label, set_df, formula)
        }

        # Collect the results
        if (counter2 == 1) emm_df <- rbind.fill(emm_df, out$out_df)
        if (counter2 != 1) emm_df2 <- rbind.fill(emm_df2, out$out_df)
        if (counter == 1) res <- rbind(res, out$results)
        if (counter != 1) res2 <- rbind(res2, out$results)

        # Progress bar
        cat(paste0('\rProgress: ', progress, ' of ', length(unique(data$Set)),
          ' sets done for ', new_label, '.\r'))
        progress <- progress + 1
      } # all sets are done

      cat('\n') # new progress bar
    
      # Combine the genotype estimated marginal means of the transformed labels
      if (counter2 != 1) {
        emm_df <- left_join(emm_df, emm_df2, by=join_cols, keep=F)
      }
      counter2 <- counter2 + 1
    } # only if new_label exists
  } # label and all transformed labels are modeled

  # Combine the epistasis results for all transformed labels to the original label
  if (counter == 1) {
    colnames(res) <- c('Set', 'e_est', 'lowerCI', 'upperCI', 'rsquared',
      'pval_e', 'Label', 'Term')
  }
  if (counter != 1) {
    colnames(res2) <- c('Set', 'e_est', 'lowerCI', 'upperCI', 'rsquared',
      'pval_e', 'Label', 'Term')
    res <- rbind.fill(res, res2)
  }

  # Save the genotype estimated marginal  means
  emm_df <- emm_df[, !grepl("\\.x$|\\.y$", colnames(emm_df))]
  emm_df <- emm_df[, !(colnames(emm_df) %in% c('pop_mean'))]
  write.table(emm_df, paste0(
    'data/20240923_melissa_ara_data/corrected_data/fitness_data_for_Kenia_09232024_',
    label, '_emmeans.tsv'),
    row.names=F, quote=F, sep='\t')
  remove(emm_df, emm_df2, res2) # clear memory

  counter <- counter + 1
}

# Determine direction of epistasis
df_results <- res %>% mutate(
  Epistasis_Direction = case_when(lowerCI < 0 & upperCI < 0  ~ 'Negative',
                                  lowerCI > 0 & upperCI > 0 ~ 'Positive',
                                  lowerCI <=  0 & upperCI >= 0 ~ 'Not Detected'))
table(df_results$Epistasis_Direction)
table(df_results$Label)

# Save the epistasis results
write.csv(df_results,
  paste0('data/20240923_melissa_ara_data/corrected_data/fitness_data_for_Kenia_09232024_epistasis_linear.csv'),
  row.names=F, quote=F)
