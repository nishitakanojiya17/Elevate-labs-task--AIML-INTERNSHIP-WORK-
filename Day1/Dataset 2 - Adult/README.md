# Data Cleaning and Preprocessing — Dat 1 task

## Project Overview

This project focuses on cleaning and preprocessing the Adult Census Income dataset for machine learning. The workflow includes inspecting the data, handling missing values, encoding categorical variables, scaling numerical features, and identifying outliers.

## Dataset

- **Name:** Adult Census Income
- **Source:** UCI Machine Learning Repository
- **File:** `dataset/adult.csv`
- **Target:** Income

The dataset contains demographic and employment-related attributes used to predict whether an individual's annual income exceeds $50,000.

## Project Structure

```text
Day-1-task-Elevate-Lab---Nishita-kanojiya/
├── dataset/
│   └── adult.csv
├── notebooks/
│   └── data_preprocessing.ipynb
├── src/
│   └── preprocessing.py
├── outputs/
├── README.md
├── requirements.txt
└── .gitignore
```

## Preprocessing Workflow

1. Load and inspect the dataset.
2. Identify missing values and inconsistent entries.
3. Handle missing values in numerical and categorical columns.
4. Encode categorical features.
5. Standardize numerical features.
6. Detect and remove outliers where appropriate.
7. Save the processed dataset and document the results.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run

Open `notebooks/data_preprocessing.ipynb` in VS Code and run the notebook cells in order.

## Tools

- Python
- pandas
- NumPy
- scikit-learn
- Jupyter
- ucimlrepo
