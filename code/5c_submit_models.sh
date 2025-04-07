#!/bin/bash

conda activate py310
cd ara-kinase-prediction

''' Run Cusack 2021 dataset through the lab RF pipeline to determine if my
XGBoost and AutoGluon classification code are working correctly.'''

## XGBoost
# Original Dataset 4 from Cusack 2021
python code/5a_classification.py \
	-X data/2021_cusack_data/Dataset_4.txt \
	-y_name Class -cl_list negative,positive \
	-test data/2021_cusack_data/Dataset_4_test_instances.txt \
	-save data/2021_cusack_data/output_clf \
	-prefix xgb_clf_original_Dataset_4 \
	-alg xgboost -bal y -n_bal 100 \
	-tag debug_xgb_clf_with_cusack_2021 \
	-ht kfold -fold 5 -n 10 -feat all -plot f

# Imputed Dataset 4 from features we re-generated
python code/5a_classification.py \
	-X data/2021_cusack_data/Dataset_4_Features/Imputed_Dataset_4_final_table.csv \
	-y_name Class -cl_list negative,positive \
	-test data/2021_cusack_data/Dataset_4_test_instances.txt \
	-save data/2021_cusack_data/Dataset_4_Features/output_clf \
	-prefix xgb_clf_imputed_Dataset_4 \
	-alg xgboost -bal y -n_bal 100 \
	-tag debug_xgb_clf_with_cusack_2021 \
	-ht kfold -fold 5 -n 10 -feat all -plot f

## RandomForest (these were submitted in HPCC)
python /home/seguraab/external_software/ML-Pipeline/ML_classification.py \
	-df data/2021_cusack_data/Dataset_4_Features/Imputed_Dataset_4_final_table.csv \
	-y_name Class -pos positive -apply test -cl_train positive,negative \
	-alg RF -n_jobs 14 -b 100 -cv_num 5 -x_norm t -gs_reps 10 -cm t \
	-save rf_clf_imputed_Dataset_4 -tag debug_xgb_clf_with_cusack_2021_imputed \
	-plots f

python /home/seguraab/external_software/ML-Pipeline/ML_classification.py \
	-df data/2021_cusack_data/Dataset_4.txt \
	-y_name Class -pos positive -apply test -cl_train positive,negative \
	-alg RF -n_jobs 14 -b 100 -cv_num 5 -x_norm t -gs_reps 10 -cm t \
	-save rf_clf_og_Dataset_4 -tag debug_xgb_clf_with_cusack_2021 \
	-plots f

# :'''Preliminary classification models with the 603 features that contained no ############################# re-run these, 9/18/2024 (bc I added normalization)
# missing data. Hyperparameter tuning was implemented with two methods:
# Stratified K-Fold Cross-Validation and Leave-One-Out Cross-Validation.
# '''
# # Submission: nohup bash code/5c_submit_models.sh >> logs/5c_prelim_xgb_class_603_features.txt

# for i in {0..9}; do
# 	# Stratified K-Fold Cross-Validation Hyperparameter Tuning Method
# 	python code/5a_xgb_classification.py \
# 		-X data/Features/Table_features_kept_kinase_prediction_train.csv \
# 		-y_name Y \
# 		-test data/test_sets_clf/test_ara_m_fold_${i}.txt \
# 		-save output_clf/ara_m_kfold_ht_603_feats \
# 		-prefix ara_m_kfold_ht_test_${i} \
# 		-tag ara_m_kfold_ht_test_${i} \
# 		-ht kfold -fold 5 -n 10 -feat all -plot f
	
# 	# Leave-One-Out Cross-Validation Hyperparameter Tuning Method
# 	python code/5a_xgb_classification.py \
# 		-X data/Features/Table_features_kept_kinase_prediction_train.csv \
# 		-y_name Y \
# 		-test data/test_sets_clf/test_ara_m_fold_${i}.txt \
# 		-save output_clf/ara_m_loo_ht_603_feats \
# 		-prefix ara_m_kfold_ht_test_${i} \
# 		-tag ara_m_kfold_ht_test_${i} \
# 		-ht kfold -fold 5 -n 10 -feat all -plot f
# done


:'''Preliminary classification models with the imputed features. Each table has ############################# re-run these, 9/18/2024
2259 features with <30% missing data imputed with K-Nearest Neighbors method.
Hyperparameter tuning was implemented with two methods:
Stratified K-Fold Cross-Validation and Leave-One-Out Cross-Validation.
'''
# Submission: nohup bash code/5c_submit_models.sh >> logs/5c_prelim_xgb_class_2259_imputed_features.txt

# I re-ran the following code on 9/XX/2024
for i in {0..9}; do
	# AutoGluon Implementation
	python code/5a_classification.py \
		-X data/Features/imputed_features/Table_features_imputed_kinase_prediction_train_test_ara_m_fold_${i}.csv \
		-y_name Y \
		-test data/test_sets_clf/test_ara_m_fold_${i}.txt \
		-save /home/seguraab/ara-kinase-prediction/output_clf/ara_m_autogluon_2259_imp_feats \
		-prefix ara_m_autogluon_test_${i} \
		-alg autogluon -bal y -n_bal 15 \
		-tag ara_m_autogluon_test_${i}
done

# I re-ran the following code on 9/XX/2024
for i in {0..9}; do
	# Stratified K-Fold Cross-Validation Hyperparameter Tuning Method
	python code/5a_classification.py \
		-X data/Features/imputed_features/Table_features_imputed_kinase_prediction_train_test_ara_m_fold_${i}.csv \
		-y_name Y \
		-test data/test_sets_clf/test_ara_m_fold_${i}.txt \
		-save /home/seguraab/ara-kinase-prediction/output_clf/ara_m_kfold_ht_2259_imp_feats \
		-prefix ara_m_kfold_ht_2259_imp_feats_test_${i} \
		-alg xgboost -bal y -n_bal 15 \
		-tag ara_m_kfold_ht_2259_imp_feats_test_${i} \
		-ht kfold -fold 5 -n 10 -feat all -plot f
done

# I re-ran the following code on 9/XX/2024
:'''Feature selection on the 2259 imputed features. Feature selection is performed
using RandomForestClassifier with the Gini impurity criterion.
'''
for i in {0..9}; do
	# Stratified K-Fold Cross-Validation Hyperparameter Tuning Method
	python code/5a_classification.py \
		-X data/Features/imputed_features/Table_features_imputed_kinase_prediction_train_test_ara_m_fold_${i}.csv \
		-y_name Y \
		-test data/test_sets_clf/test_ara_m_fold_${i}.txt \
		-save /home/seguraab/ara-kinase-prediction/output_clf/ara_m_kfold_ht_2259_imp_feats_fs \
		-prefix ara_m_kfold_ht_2259_imp_feats_test_${i} \
		-tag ara_m_kfold_ht_2259_imp_feats_test_${i} \
		-ht kfold -fold 5 -n 10 -feat all -plot f -bal y -n_bal 15 -alg xgboost \
		-fs y -start 75 -stop 2259 -step 75 -write y -type gini
done


:'''Preliminary regression models with the imputed features. Each table has
2259 features with <30% missing data imputed with K-Nearest Neighbors method.
Hyperparameter tuning was implemented with two methods:
Stratified K-Fold Cross-Validation and Leave-One-Out Cross-Validation.
'''
labels=(TSC_plog10_emmean.epi_min TSC_plog10_emmean.epi_product
TSC_plog10_emmean.epi_additive TSC_plog10_emmean.epi_log2_mani
TSC_plog10_emmean.epi_mean TSC_plog10_emmean.epi_max
TSC_plog10_emmean.epi_log2_additive
TSC_plog10_emmean.epi_log2_difference TSC_plus1_emmean.epi_min
TSC_plus1_emmean.epi_product TSC_plus1_emmean.epi_additive
TSC_plus1_emmean.epi_log2_mani TSC_plus1_emmean.epi_mean
TSC_plus1_emmean.epi_max TSC_plus1_emmean.epi_log2_additive
TSC_plus1_emmean.epi_log2_difference
TSC_plus1_log10_emmean.epi_min TSC_plus1_log10_emmean.epi_product
TSC_plus1_log10_emmean.epi_additive
TSC_plus1_log10_emmean.epi_log2_mani
TSC_plus1_log10_emmean.epi_mean TSC_plus1_log10_emmean.epi_max
TSC_plus1_log10_emmean.epi_log2_additive
TSC_plus1_log10_emmean.epi_log2_difference)

for i in {0..9}; do
	for label in "${labels[@]}"; do
		# Stratified K-Fold Cross-Validation Hyperparameter Tuning Method
		python code/5b_regression.py \
			-X data/Features/imputed_features/Table_features_imputed_kinase_prediction_train_test_ara_m_fold_${i}.csv \
			-drop Y \
			-Y data/20240923_melissa_ara_data/corrected_data/fitness_data_for_Kenia_09232024_TSC_emmeans_epistasis_regression_labels.csv \
			-y_name $label \
			-test data/test_sets_clf/20240725_ara_m_test_sets_old_clf_label/test_ara_m_fold_${i}.txt \
			-save /home/seguraab/ara-kinase-prediction/output_reg/ara_m_kfold_ht_2259_imp_feats \
			-prefix ara_m_kfold_ht_2259_imp_feats_test_${label}_${i} \
			-tag ara_m_kfold_ht_2259_imp_feats_test_${label}_${i} \
			-alg xgboost -ht kfold -fold 5 -n 10 -feat all -plot f
	done
done
