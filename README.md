# Heart Disease MLOps Project

## Overview

This project implements an end-to-end machine learning and MLOps
pipeline using the UCI Heart Disease dataset.

The project covers:

1. Data acquisition and EDA
2. Feature engineering and model development
3. MLflow experiment tracking
4. Model packaging and reproducibility
5. Automated testing and CI/CD
6. Docker containerization
7. Kubernetes deployment
8. Monitoring and logging
9. Documentation and reporting

## Dataset

The project uses the Heart Disease dataset from the
UCI Machine Learning Repository.

Dataset ID: 45

Source:

https://archive.ics.uci.edu/dataset/45/heart+disease

## Stage 1

Stage 1 performs:

- Dataset acquisition
- Data validation
- Missing-value marker conversion
- Data type normalization
- Binary target creation
- Exploratory data analysis
- EDA visualization generation

## Installation

Create a virtual environment:

```bash
python -m venv .venv

## Stage 2

                    heart_disease_clean.csv
                              │
                              ▼
                    Load cleaned dataset
                              │
                              ▼
                     Separate X and y
                              │
                              ▼
                  Stratified Train/Test Split
                         80% / 20%
                         │       │
                    Training    Test
                       │          │
                       ▼          │
             5-Fold Stratified CV │
                       │          │
              ┌────────┴────────┐ │
              ▼                 ▼ │
       Logistic Regression   Random Forest
              │                 │
              ▼                 ▼
       Hyperparameter       Hyperparameter
          tuning               tuning
              │                 │
              └────────┬────────┘
                       ▼
              Compare CV ROC-AUC
                       │
                       ▼
                 Best Model
                       │
                       ▼
            Evaluate ONCE on test set
                       │
                       ▼
             Save complete pipeline
                       │
                       ▼
                MLflow artifacts