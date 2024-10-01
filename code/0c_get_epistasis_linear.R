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
    return(c(set, e_est, lowerCI, upperCI, rsquared, pval_e))
    }, error = function(e) {
      warning("An error occurred:", conditionMessage(e))
      return(rep(NA, 6)) # Return a vector of names
    })

  return(result)
}

# Calculate the epistasis values for each plant set and fitness label
labels <- c('GN', 'PG', 'DTB', 'LN', 'DTF', 'SN', 'WO', 'FN', 'SPF', 'TSC', 'SH')
counter <- 1
for (label in labels){
    print(paste0('LABEL ', label, ' -----------------------------------------'))

    # Collect epistasis values for each label
    if (counter == 1) res <- data.frame()
    if (counter != 1) res2 <- data.frame()

    # Run linear regression per set
    for (set in unique(data$Set)) {
        # print(paste0('Set ', set))
        set_df <- data[data$Set == set, c("Set", "Flat", "Column", "Row",
                      "Number", "Type", "Genotype", "Subline", "MA", "MB", "DM",
                      paste0(label, '_corrected'))] # set data

        # Remove rows with missing values in label
        set_df <- set_df[!is.na(set_df[,paste0(label, '_corrected')]),]
        if (nrow(set_df) == 0) next

        # Calculate the epistasis value
        pop_mean <- mean(set_df[set_df$Genotype == 'WT', paste0(label, '_corrected')])
        set_df$pop_mean <- pop_mean # population mean
        out <- get_epi_stats(set, set_df, paste0(label, '_corrected'))

        # Collect the results
        if (counter == 1) res <- rbind(res, out)
        if (counter != 1) res2 <- rbind(res2, out)
    }
    
    if (counter == 1) {
      colnames(res) <- c('Set', 'e_est', 'lowerCI', 'upperCI', 'rsquared', 'pval_e')
      res$Label <- labels[1]
    }

    if (counter != 1) {
        print(dim(res))
        print(dim(res2))
        colnames(res2) <- c('Set', 'e_est', 'lowerCI', 'upperCI', 'rsquared', 'pval_e')
        res2$Label <- labels[counter]
        res <- rbind.fill(res, res2)
    }
    counter <- counter + 1
}

# Determine direction of epistasis
df_results <- res %>% mutate(
  Epistasis_Direction = case_when(lowerCI < 0 & upperCI < 0  ~ 'Negative',
                                  lowerCI > 0 & upperCI > 0 ~ 'Positive',
                                  lowerCI <=  0 & upperCI >= 0 ~ 'Not Detected'))
table(df_results$Epistasis_Direction)

# Save the results
write.csv(df_results,
  paste0('data/20240917_melissa_ara_data/corrected_data/fitness_data_for_Kenia_09172024_corrected_epistasis_linear.csv'),
  row.names=F, quote=F)
