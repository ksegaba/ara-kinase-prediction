#! /usr/bin/env python3
'''Feature selection using SHAP values from a pre-trained Random Forest model.
Features are selected based on their absolute SHAP values using percentile
thresholds, and feature sets are generated for running Random Forest models.
This script also generates a SLURM sbatch file to run the Random Forest models
using the generated feature sets.

Arguments:
    --model_path: Path to the pre-trained Random Forest model.
    --data_path: Path to the dataset CSV file.
    --output_dir: Directory to save the generated feature sets.
    --sbatch_name: Name to save the generated SLURM sbatch file.
    --tag: Prefix for the RF model output files.

Dependencies:
    - Conda environment: conda activate py310
    - shap version: 0.47.2
    - scikit-learn version: 1.5.0 (AutoGluon requires 1.4.0)
'''

import joblib
import shap
import os
import argparse
import numpy as np
import pandas as pd
import datatable as dt


def load_rf_model(model_path):
    # 1) Load a pre-trained Random Forest model
    model = joblib.load(model_path)  # Use joblib to load the model
    return model


def calculate_shap_values(model, X_array, X):
    # 2) Calculate SHAP values for each feature for the positive class
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_array)  # Positive class is at idx 1
    shap_values = pd.DataFrame(
        shap_values[:, :, 1], columns=X.columns, index=X.index)
    return shap_values


def generate_feature_sets(shap_values, output_dir):
    '''3) Generate feature sets for running feature selection
    Outputs:
      - feature_set_above_p{percentile}.txt: list of features above the
        threshold for each percentile if the threshold is unique.'''
    #
    # Average absolute SHAP values
    mean_abs_shap = np.mean(np.abs(shap_values),
                            axis=0).sort_values(ascending=False)
    #
    # Determine unique thresholds for feature selection
    unique_thresholds = set()
    feature_sets = {}
    percentiles = np.arange(0.1, 1.0, 0.1)
    #
    for percentile in percentiles:
        threshold = mean_abs_shap.quantile(round(percentile, 1))
        #
        if threshold not in unique_thresholds:
            unique_thresholds.add(threshold)
            #
            # Select features above the threshold
            feature_sets[round(percentile, 1)] = mean_abs_shap[
                mean_abs_shap >= threshold].index
            #
            # Save the feature set to a file
            feature_set_path = os.path.join(
                output_dir, f'feature_set_above_p{round(percentile, 1)}.txt')
            pd.DataFrame(feature_sets[round(percentile, 1)]).to_csv(
                feature_set_path, index=False, header=False)
            #
    return feature_sets


def generate_sbatch_file(output_dir, data_path, sbatch_name, tag, feature_sets):
    '''4) Generate a SLURM sbatch file to run RF models with different feature
    sets on the HPC cluster.'''
    #
    with open(os.path.join(output_dir, sbatch_name), 'w') as f:
        f.write("#!/bin/bash --login\n")
        f.write("#SBATCH --job-name=rf_feature_selection\n")
        f.write("#SBATCH --output=rf_feature_selection_%A_%a.out\n")
        f.write("#SBATCH --error=rf_feature_selection_%A_%a.err\n")
        f.write("#SBATCH --array=0-{}\n".format(len(feature_sets) - 1))
        f.write("#SBATCH --time=04:00:00\n")
        f.write("#SBATCH --nodes=1\n")
        f.write("#SBATCH --ntasks=2\n")
        f.write("#SBATCH --mem=3G\n")
        f.write("#SBATCH --cpus-per-task=12\n\n")
        f.write("module purge\n")
        f.write("conda activate ml-pipe1.5\n\n")
        f.write(f"cd {output_dir}\n")
        f.write(
            f"FEATURE_SETS=($(ls feature_set_above_p*.txt))\n\n")
        f.write(
            "python /mnt/home/seguraab/Shiu_Lab/ML-Pipeline/ML_classification.py \\\n")
        f.write(f"\t-df {data_path} -sep ,\\\n")
        f.write("\t-y_name Class -pos positive -cl_train positive,negative \\\n")
        f.write("\t-feat ${FEATURE_SETS[${SLURM_ARRAY_TASK_ID}]} \\\n")
        f.write(
            "\t-alg RF -n_jobs 12 -b 100 -cv_num 10 -x_norm t -gs_reps 10 -cm t \\\n")
        f.write(f"\t-save {tag}")
        f.write("_${FEATURE_SETS[${SLURM_ARRAY_TASK_ID}]} \\\n")
        f.write("\t-plots f\n\n")
        f.write("conda deactivate\n\nscontrol show job $SLURM_JOB_ID\n")


# Example usage
if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(
        description="Feature selection using SHAP values and SLURM job generation.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pre-trained Random Forest model.")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the dataset CSV file.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the generated feature sets and SLURM script.")
    parser.add_argument("--sbatch_name", type=str, required=True,
                        help="Name to save as the generated SLURM sbatch file.")
    parser.add_argument("--tag", type=str, default='',
                        help="Prefix for the RF model output files.")

    args = parser.parse_args()

    # Paths
    model_path = args.model_path
    data_path = args.data_path
    output_dir = args.output_dir
    sbatch_name = args.sbatch_name
    tag = args.tag

    # Arguments for example usage
    # model_path = "/home/seguraab/ara-kinase-prediction/data/2021_cusack_data/Dataset_4_Features/output_clf/RF/rf_clf_imputed_Dataset_4by_bin_cat_cont_type_models.pkl"
    # data_path = "/home/seguraab/ara-kinase-prediction/data/2021_cusack_data/Dataset_4_Features/Imputed_Dataset_4_X_train_final_table_by_bin_cat_cont_type.csv"
    # output_dir = "/home/seguraab/ara-kinase-prediction/data/2021_cusack_data/Dataset_4_Features/output_clf/RF_FS"
    # sbatch_name = "run_rf_feature_selection_models.sb"
    # tag = "rf_fs_clf_imputed_Dataset_4by_bin_cat_cont_type"

    # Load data and model
    data = dt.fread(data_path).to_pandas()
    data = data.set_index(data.columns[0])

    # Replace "target" with your target column name
    X = data.drop(columns=["Class"])
    del data

    # Load the pre-trained Random Forest model
    model = load_rf_model(model_path)

    # Calculate SHAP values
    shap_values = calculate_shap_values(model, X.to_numpy(), X)

    # Generate feature sets
    feature_sets = generate_feature_sets(shap_values, output_dir)

    # Generate SLURM sbatch file
    generate_sbatch_file(output_dir, data_path, sbatch_name, tag, feature_sets)
