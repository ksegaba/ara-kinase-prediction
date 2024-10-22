# Description: Calculate epistasis values for each plant set and fitness trait

library(readxl)
library(tidyr)
library(plyr)
library(dplyr)
library(CR2) # for ncvMLM
library(lmtest) # for bptest and dwtest
library(lmerTest)
library(performance) # for lmer epistasis stats and check_autocorrelation
library(emmeans)

# Prepare fitness data
data <- read_excel(
  'data/20240923_melissa_ara_data/fitness_data_for_Kenia_09232024.xlsx',
  sheet='with_border_cells')

data$MA <- 0
data[data$Genotype == 'MA', 'MA'] <- 1
data$MB <- 0
data[data$Genotype == 'MB', 'MB'] <- 1
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

check_assumptions <- function(model, lmer=TRUE){
  # Check assumptions of the linear model
  # 1. Normality of the error distribution (Shapiro-Wilk test)
  # 2. Homoscedasticity (constant variance) of the errors (Breush-Pagan test)
  # 3. Independence of the errors (Durbin-Watson test)
  # 4. Linearity and additivity of the relationship between dependent and independent variables (did not check)
  # Note: I'm not sure why, but the function errors out when an lmer model is 
  # passed, causing only NA values to be returned. When I run each line by itself,
  # it works fine. 
  # In value[[3L]](cond) :
  # An error occurred:Unable to extract deviance function from model fit

  assumptions <- tryCatch({
    # Obtain model residuals
    residuals <- resid(model)
    fitted_vals <- fitted(model)

    if (lmer == TRUE) {
      sw_test <- shapiro.test(residuals) # 1. Normality of residuals
      bp_test <- ncvMLM(model, bp=T) # 2. Homoscedasticity
      dw_test <- check_autocorrelation(model, nsim=10000)[[1]] # 3. Independence of residuals

      return(list(sw_test$statistic[[1]], sw_test$p.value, 'NA',
        bp_test, 'NA', dw_test))
    } else {
      sw_test <- shapiro.test(residuals)
      bp_test <- bptest(model)
      dw_test <- dwtest(model)

      return(list(
        sw_test$statistic[[1]], sw_test$p.value,
        bp_test$statistic[[1]], bp_test$p.value[[1]],
        dw_test$statistic[[1]], dw_test$p.value))
    }
  }, error = function(e) {
    warning('An error occurred:', conditionMessage(e))
    return(rep(NA, 6)) # Return a vector of NAs
  })

  return(assumptions)
}

get_epi_stats <- function(set, label, set_df, formula, lmer=TRUE){
  # Calculate the epistasis values and genotype estimated marginal means of
  # the label of interest with a linear model
  
  result <- tryCatch({
    if (lmer == TRUE){
      model <- lmer(formula, data=set_df)

      # Extract the epistasis stats of the interaction term
      e_est <- as.numeric(fixef(model)['MA:MB'])
      lowerCI <- confint(model, level=0.95)['MA:MB',1]
      upperCI <- confint(model, level=0.95)['MA:MB',2]
      r2_model <- r2_nakagawa(model, tolerance=1e-1000)
      rsquared <- as.numeric(r2_model[[2]]) # Marginal R-squared takes into account only the variance of the fixed effects
      pval_e <- as.numeric(coef(summary(model))[,'Pr(>|t|)']['MA:MB'])

    } else {
      model <- lm(formula, data=set_df)

      # Extract the epistasis stats of the interaction term
      e_est <- as.numeric(coef(model)['MA:MB'])
      lowerCI <- confint(model)['MA:MB',1]
      upperCI <- confint(model)['MA:MB',2]
      rsquared <- summary(model)$adj.r.squared
      pval_e <- as.numeric(summary(model)$coefficients[,4]['MA:MB'])
    }
    
    # # Check assumptions (not working for an unknown reason)
    # assumptions <- check_assumptions(model, lmer)
    
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
    emm <- rename_with(emm, ~ paste0(label, '_', .))
    out_df <- left_join(set_df, emm, by=c('Genotype'=paste0(label, '_Genotype')))
    out_df <- out_df[, !(colnames(out_df) %in% c(paste0(label, '_MA'), paste0(label, '_MB')))]

    # Return stats
    return(list(#assumptions=assumptions,
      results=c(set, e_est, lowerCI, upperCI, rsquared, pval_e, label, 'MA:MB'),
      out_df=out_df, model=model))
  }, error = function(e) {
    warning('An error occurred:', conditionMessage(e))
    return(list(#assumptions=rep(NA, 6),
      results=c(set, rep(NA, 5), label, 'MA:MB'), out_df=set_df, model=NULL))
  })
  
  return(result)
}

# Calculate the epistasis values for each plant set and fitness label
labels <- c('PG', 'DTB', 'LN', 'DTF', 'SN', 'WO', 'FN', 'SPF', 'TSC', 'SH')
counter <- 1
for (label in labels){
  print(paste0('LABEL ', label, ' -----------------------------------------'))
  
  join_cols <- c('Set', 'Flat', 'Column', 'Row', 'Number', 'Type', 'Genotype',
    'Subline', 'MA', 'MB', label)

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
          set_df <- data[data$Set == set, c('Set', 'Flat', 'Column', 'Row',
            'Number', 'Type', 'Genotype', 'Subline', 'MA', 'MB',
            label, paste0(label, '_plus1'), paste0(label, '_plus1_log10'),
            paste0(label, '_plog10'))] # set data
        } else {
            set_df <- data[data$Set == set, c('Set', 'Flat', 'Column', 'Row',
              'Number', 'Type', 'Genotype', 'Subline', 'MA', 'MB',
              label, paste0(label, '_log10'), paste0(label, '_plog10'))]
        }

        set_df <- set_df[!is.na(set_df[,label]),] # Remove rows with missing values in label
        if (nrow(set_df) == 0) next # skip sets with no data

        # Define the linear equation
        if (length(unique(set_df$Flat)) > 1) {
          formula <- paste0(new_label, ' ~ MA + MB + MA:MB + (1|Flat)')
          lmer <- TRUE
        } else {
          formula <- paste0(new_label, ' ~ MA + MB + MA:MB')
          lmer <- FALSE
        }

        # Calculate the epistasis statistics
        if (endsWith(new_label, '_log10')) { # deal with infinite values
          set_df_sub <- set_df[!is.infinite(set_df[[new_label]]),]
          if (nrow(set_df_sub) == 0) {
            next
          } else {
            if (length(set_df_sub$Genotype) < 4) {
              next
            } else {
              out <- get_epi_stats(set, new_label, set_df_sub, formula, lmer)
            }
          }
        } else {
          out <- get_epi_stats(set, new_label, set_df, formula, lmer)
        }

        # Check assumptions
        if (is.null(out$model) == FALSE) {
          if (lmer == TRUE) {
            residuals <- resid(out$model)
            fitted_vals <- fitted(out$model)
            sw_test <- shapiro.test(residuals) # 1. Normality of residuals
            bp_test <- ncvMLM(out$model, bp=T) # 2. Homoscedasticity
            dw_test <- check_autocorrelation(out$model, nsim=10000)[[1]] # 3. Independence of residuals
            assumptions <- list(sw_test$statistic[[1]], sw_test$p.value, NA,
              bp_test, NA, dw_test)

          } else {
            residuals <- resid(out$model)
            fitted_vals <- fitted(out$model)
            sw_test <- shapiro.test(residuals)
            bp_test <- bptest(out$model)
            dw_test <- dwtest(out$model)
            assumptions <- list(
              sw_test$statistic[[1]], sw_test$p.value,
              bp_test$statistic[[1]], bp_test$p.value[[1]],
              dw_test$statistic[[1]], dw_test$p.value)
          }

          # Collect the results
          if (counter2 == 1) emm_df <- rbind.fill(emm_df, out$out_df)
          if (counter2 != 1) emm_df2 <- rbind.fill(emm_df2, out$out_df)
          if (counter == 1) res <- rbind(res, c(out$results, assumptions))
          if (counter != 1) res2 <- rbind(res2, c(out$results, assumptions))
        }

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
      'pval_e', 'Label', 'Term', 'sw_test_stat', 'sw_test_pval', 'bp_test_stat',
      'bp_test_pval', 'dw_test_stat', 'dw_test_pval')
  }
  if (counter != 1) {
    colnames(res2) <- c('Set', 'e_est', 'lowerCI', 'upperCI', 'rsquared',
      'pval_e', 'Label', 'Term', 'sw_test_stat', 'sw_test_pval', 'bp_test_stat',
      'bp_test_pval', 'dw_test_stat', 'dw_test_pval')
    res <- rbind.fill(res, res2)
  }

  # Save the genotype estimated marginal  means
  emm_df <- emm_df[, !grepl('\\.x$|\\.y$', colnames(emm_df))]
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
