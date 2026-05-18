# ML Model Comparison Tool

A Python-based Machine Learning command-line tool that allows users to compare multiple ML models on any CSV dataset for both **Classification** and **Regression** tasks.

The tool automates preprocessing, training, evaluation, and reporting in a single pipeline.

---

# Project Overview

This project loads a dataset from a CSV file, preprocesses the data automatically, trains multiple machine learning models, evaluates them, and then selects the best performing model.

It also generates a final report file with all results.

---

# Features

## Dataset Handling
- Load any CSV dataset
- Automatically detect features and target column
- Handle missing values in both numerical and categorical data
- Drop high-cardinality categorical columns

## Preprocessing
- Fill missing numerical values with mean
- Fill missing categorical values with mode
- One-hot encoding for categorical features
- Label encoding for classification targets
- Feature scaling using StandardScaler

## Task Support
- Classification tasks
- Regression tasks

---

# Machine Learning Models

## Classification Models
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree Classifier
- Random Forest Classifier

## Regression Models
- Linear Regression
- Support Vector Regressor (SVR)
- Decision Tree Regressor
- Random Forest Regressor

---

# Evaluation Metrics

## Classification
- Accuracy Score
- F1 Score (weighted)

## Regression
- Mean Squared Error (MSE)
- R² Score

---

# Workflow

## Step 1: Load Dataset
The dataset is loaded from a CSV file provided by the user.

## Step 2: Data Cleaning
- Remove rows with missing target values
- Handle missing values in features

## Step 3: Feature Processing
- Encode categorical variables
- Apply one-hot encoding
- Scale features using StandardScaler

## Step 4: Train/Test Split
- 80% training data
- 20% testing data

## Step 5: Model Training
Multiple machine learning models are trained automatically.

## Step 6: Model Evaluation
Each model is evaluated using appropriate metrics.

## Step 7: Results Comparison
- Display all model results in a table
- Select the best performing model

## Step 8: Report Generation
A text file is created containing:
- Dataset path
- Task type
- Model comparison results
- Best model

---

# Command Line Usage

Run the program using:

## Classification
```bash
python main.py dataset.csv classification

## Regression

```bash
python main.py dataset.csv regression

# Output

After execution, the program will:

- Train multiple models  
- Print performance results  
- Show the best model  
- Save a report file (`model_report.txt`)  

---

# Example Use Case

A user provides a dataset and selects a task type.

The system will:

- Automatically preprocess the dataset  
- Train 4 different models  
- Compare their performance  
- Output the best model  

---

# Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Argparse  

---


---

# Author

## Saja

Master’s Student in Artificial Intelligence  
