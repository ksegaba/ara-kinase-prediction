#!/bin/bash

conda activate py310
cd ara-kinase-prediction

# :'''Preliminary classification models with the 603 features that contained no
# missing data. Hyperparameter tuning was implemented with two methods:
# Stratified K-Fold Cross-Validation and Leave-One-Out Cross-Validation.
# '''
# # Submission: nohup bash code/5c_submit_models.sh >> logs/5c_prelim_xgb_class_603_features.txt

# for i in {0..9}; do
# 	# Stratified K-Fold Cross-Validation Hyperparameter Tuning Method
# 	python code/5a_xgb_classification.py \
# 		-X data/Features/Table_features_kept_kinase_prediction_train.csv \
# 		-y_name Y \
# 		-test data/test_ara_m_fold_${i}.txt \
# 		-save output/ara_m_kfold_ht_603_feats \
# 		-prefix ara_m_kfold_ht_test_${i} \
# 		-tag ara_m_kfold_ht_test_${i} \
# 		-ht kfold -fold 5 -n 10 -feat all -plot f
	
# 	# Leave-One-Out Cross-Validation Hyperparameter Tuning Method
# 	python code/5a_xgb_classification.py \
# 		-X data/Features/Table_features_kept_kinase_prediction_train.csv \
# 		-y_name Y \
# 		-test data/test_ara_m_fold_${i}.txt \
# 		-save output/ara_m_loo_ht_603_feats \
# 		-prefix ara_m_kfold_ht_test_${i} \
# 		-tag ara_m_kfold_ht_test_${i} \
# 		-ht kfold -fold 5 -n 10 -feat all -plot f
# done


:'''Preliminary classification models with the imputed features. Each table has
2259 features with <30% missing data imputed with K-Nearest Neighbors method.
Hyperparameter tuning was implemented with two methods:
Stratified K-Fold Cross-Validation and Leave-One-Out Cross-Validation.
'''
# Submission: nohup bash code/5c_submit_models.sh >> logs/5c_prelim_xgb_class_2259_imputed_features.txt

for i in {0..9}; do
	# AutoGluon Implementation
	python code/5a_autogluon_classification.py \
		data/Features/imputed_features/Table_features_imputed_kinase_prediction_train_test_ara_m_fold_${i}.csv \
		data/test_ara_m_fold_${i}.txt \
		Y \
		output/ara_m_autogluon_2259_imp_feats \
		ara_m_autogluon_test_${i}
done

for i in {0..9}; do
	# Stratified K-Fold Cross-Validation Hyperparameter Tuning Method
	python code/5a_classification.py \
		-X data/Features/imputed_features/Table_features_imputed_kinase_prediction_train_test_ara_m_fold_${i}.csv \
		-y_name Y \
		-test data/test_ara_m_fold_${i}.txt \
		-save output/ara_m_kfold_ht_2259_imp_feats \
		-prefix ara_m_kfold_ht_test_${i} \
		-tag ara_m_kfold_ht_test_${i} \
		-ht kfold -fold 5 -n 10 -feat all -plot f
done

for i in {0..9}; do
	# Leave-One-Out Cross-Validation Hyperparameter Tuning Method
	python code/5a_classification.py \
		-X data/Features/imputed_features/Table_features_imputed_kinase_prediction_train_test_ara_m_fold_${i}.csv \
		-y_name Y \
		-test data/test_ara_m_fold_${i}.txt \
		-save output/ara_m_loo_ht_2259_imp_feats \
		-prefix ara_m_loo_ht_test_${i} \
		-tag ara_m_loo_ht_test_${i} \
		-ht loo -fold 5 -n 10 -feat all -plot f
done
