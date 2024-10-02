# Description: Calculate epistasis values for each plant set and fitness trait

library(tidyr)
library(plyr)
library(dplyr)

# Prepare corrected fitness data
data <- read.table(
  'data/20240917_melissa_ara_data/corrected_data/fitness_data_for_Kenia_09172024_corrected.tsv',
  header=T)

data$MA <- 0
data[data$Genotype == 'MA', 'MA'] <- 1
data$MB <- 0
data[data$Genotype == 'MB', 'MB'] <- 1
data$DM <- 0
data[data$Genotype == 'DM', 'DM'] <- 1
data[data$Genotype == 'DM', 'MA'] <- 1
data[data$Genotype == 'DM', 'MB'] <- 1

# Add 1 or take the pseudolog10 of the fitness traits (some have 0 values)
# value_counts <- as.matrix(table(data$PG_corrected))
# value_counts[rownames(value_counts)==0,]

pseudolog10 <- function(x){
  # Pseudo-Logarithm of base 10, is defined for all real numbers
  return ( log((x/2) + sqrt((x/2)^2 + 1)) )/log(10)
}

labels <- c('PG_corrected', 'DTB_corrected', 'LN_corrected', 'DTF_corrected',
            'SN_corrected', 'WO_corrected', 'FN_corrected', 'SPF_corrected',
            'TSC_corrected', 'SH_corrected')

for (label in labels){
  data[, paste0(label, '_plog10')] <- pseudolog10(data[, label]) # all traits
  
  if (label %in% list('SN_corrected', 'WO_corrected', 'FN_corrected', 'TSC_corrected')){ # count traits
    data[, paste0(label, '_plus1')] <- data[, label] + 1
  }
}

get_epi_stats <- function(set, set_df, label){
  # Calculate the epistasis value
  
  formula <- paste0(label, ' ~ MA + MB + MA:MB + pop_mean')
  
  result <- tryCatch({
    model <- lm(formula, data=set_df, na.action = na.exclude)
    
    # Extract stats of interaction term
    e_est <- as.numeric(coef(model)['MA:MB'])
    lowerCI <- confint(model)['MA:MB',1]
    upperCI <- confint(model)['MA:MB',2]
    rsquared <- summary(model)$adj.r.squared
    pval_e <- as.numeric(summary(model)$coefficients[,4]['MA:MB'])
    
    # Return stats
    return(c(set, e_est, lowerCI, upperCI, rsquared, pval_e, label,  'MA:MB'))
  }, error = function(e) {
    warning("An error occurred:", conditionMessage(e))
    return(c(set, rep(NA, 5), label, 'MA:MB')) # Return a vector of names
  })

  return(result)
}

get_epi_stats2 <- function(set, set_df, label){
  # Calculate the epistasis value
  
  formula <- paste0(label, ' ~ MA + MB + DM + pop_mean')
  
  result <- tryCatch({
    model <- lm(formula, data=set_df, na.action = na.exclude)
    
    # Extract stats of interaction term
    e_est <- as.numeric(coef(model)['DM'])
    lowerCI <- confint(model)['DM',1]
    upperCI <- confint(model)['DM',2]
    rsquared <- summary(model)$adj.r.squared
    pval_e <- as.numeric(summary(model)$coefficients[,4]['DM'])
    
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
for (label in colnames(data)){
  if (endsWith(label, '_corrected') | endsWith(label, '_plog10') | endsWith(label,'_plus1')){
    print(paste0('LABEL ', label, ' -----------------------------------------'))
  
    # Collect epistasis values for each label
    if (counter == 1) res <- data.frame()
    if (counter != 1) res2 <- data.frame()

    # Run linear regression per set
    for (set in unique(data$Set)) {
        # print(paste0('Set ', set))
        set_df <- data[data$Set == set, c("Set", "Flat", "Column", "Row",
                      "Number", "Type", "Genotype", "Subline", "MA", "MB", "DM",
                      label)] # set data

        # Remove rows with missing values in label
        set_df <- set_df[!is.na(set_df[,label]),]
        if (nrow(set_df) == 0) next

        # Calculate the epistasis value
        pop_mean <- mean(set_df[set_df$Genotype == 'WT', label])
        set_df$pop_mean <- pop_mean # population mean
        out <- get_epi_stats(set, set_df, label) # has MA:MB interaction term
        out2 <- get_epi_stats2(set, set_df, label) # has DM term inplace of MA:MB
        
        # Calculate the log10 of the batch corrected label
        if (!endsWith(label, '_plog10')){
          set_df[,paste0(label, '_log10')] <- log10(set_df[,label])
          set_df <- set_df[!is.infinite(set_df[,paste0(label, '_log10')]),]
          if (nrow(set_df) == 0) next
          
          # Calculate the epistasis value
          out3 <- get_epi_stats(set, set_df, paste0(label, '_log10')) # use log10 of corrected label
          out4 <- get_epi_stats2(set, set_df, paste0(label, '_log10')) # use log10 of corrected label
        }

        # Collect the results
        if (counter == 1){
          if (!endsWith(label, '_plog10')) res <- rbind(res, out, out2, out3, out4)
          if (endsWith(label, '_plog10')) res <- rbind(res, out, out2)
        }
        if (counter != 1){
          if (!endsWith(label, '_plog10')) res2 <- rbind(res2, out, out2, out3, out4)
          if (endsWith(label, '_plog10')) res2 <- rbind(res2, out, out2)
        }
    }
    
    if (counter == 1) {
      colnames(res) <- c('Set', 'e_est', 'lowerCI', 'upperCI', 'rsquared', 'pval_e', 'Label', 'Term')
    }

    if (counter != 1) {
        print(dim(res))
        print(dim(res2))
        colnames(res2) <- c('Set', 'e_est', 'lowerCI', 'upperCI', 'rsquared', 'pval_e', 'Label', 'Term')
        res <- rbind.fill(res, res2)
    }
    counter <- counter + 1
  }
}

# Determine direction of epistasis
df_results <- res %>% mutate(
  Epistasis_Direction = case_when(lowerCI < 0 & upperCI < 0  ~ 'Negative',
                                  lowerCI > 0 & upperCI > 0 ~ 'Positive',
                                  lowerCI <=  0 & upperCI >= 0 ~ 'Not Detected'))
table(df_results$Epistasis_Direction)
table(df_results$Label)

# Save the results
write.csv(df_results,
  paste0('data/20240917_melissa_ara_data/corrected_data/fitness_data_for_Kenia_09172024_corrected_epistasis_linear.csv'),
  row.names=F, quote=F)
