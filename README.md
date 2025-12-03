# MLE-Agent Evaluation on MLEBench Lite

This repository contains the results of evaluating an autonomous machine learning agent on five MLEBench Lite datasets.  
The objective is to test real autonomy — the agent must inspect the dataset, design a pipeline, train a model, and generate a submission without any hardcoding or manual intervention.

## Overview

The agent was evaluated on the following MLEBench Lite tasks:

- siim-isic-melanoma-classification
- spooky-author-identification
- tabular-playground-series-may-2022
- text-normalization-challenge-english-language
- the-icml-2013-whale-challenge-right-whale-redux

Each dataset was evaluated using 3 random seeds, and the final metric is reported as:

Mean ± Standard Error of the Mean (SEM)

The agent autonomously:
- Inspected dataset structure
- Identified input/target columns
- Generated `generated_train.py` scripts
- Trained ML pipelines
- Produced `submission.csv` outputs

# Results Summary

| Dataset | Metric | Seeds | Score (Mean ± SEM) | Status |
|--------|--------|-------|---------------------|--------|
| Spooky Author Identification | Accuracy | 0,1,2 | 0.85 ± 0.01 | Successful |
| SIIM-ISIC Melanoma Classification | ROC-AUC | 0,1,2 | 0.6638 ± 0.0045 | Successful |
| Tabular Playground Series May 2022 | RMSE | 0,1,2 | 0.757 ± 0.002 | Successful |
| ICML 2013 Whale Challenge | constant_target | 0,1,2 | 1.000 ± 0.000 | Constant baseline |


# Experimental Setup

Hardware:
- Intel i5 quad core processor
- 16 GB RAM


Software:
- Python 3.11
- Scikit-learn, Pandas, NumPy
- Kaggle API
- MLEBench Agent Framework

Execution command:

python evaluate.py --data_dir inputs/<dataset> --output_dir outputs --seeds 0 1 2

# Dataset Preparation

Datasets were placed in:

inputs/<dataset_name>/

For Kaggle competitions:

mlebench prepare -c <competition_name>

Then copy prepared public data:

public/* → inputs/<dataset_name>

# Notes on Each Dataset

Spooky Author Identification:
- Text classification
- Agent used TF-IDF and logistic regression
- Stable performance

SIIM-ISIC Melanoma Classification:
- Tabular version
- Boosting models used
- Valid ROC-AUC scores

Tabular Playground Series (May 2022):
- Classic tabular regression
- Tree models performed consistently

Text Normalization Challenge:
- Not a standard ML task
- Contains thousands of unique target strings
- Agent unable to stratify and train

ICML Whale Challenge:
- Training set had only one class
- Agent defaulted to constant predictor
- Submission still valid

# Reproduction

Example:

python evaluate.py \
  --data_dir inputs/siim-isic-melanoma-classification \
  --output_dir outputs \
  --seeds 0 1 2

Outputs are in timestamped directories:

outputs/run_YYYYMMDD_HHMMSS_seedX/

Contents include:
- generated_train.py
- submission.csv
- stdout/stderr logs
- agent_log.jsonl

