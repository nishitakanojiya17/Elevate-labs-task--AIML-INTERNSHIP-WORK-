import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    """Load the Adult Census Income dataset."""
    return pd.read_csv(file_path)


def clean_column_names(df):
    """Remove extra spaces from column names."""
    df.columns = df.columns.str.strip()
    return df


def handle_missing_values(df):
    """Handle missing values and inconsistent entries."""

    # Replace '?' with NaN
    df = df.replace("?", np.nan)

    # Numerical columns
    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    # Fill numerical missing values with median
    for column in numerical_columns:
        df[column] = df[column].fillna(df[column].median())

    # Categorical columns
    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    # Fill categorical missing values with mode
    for column in categorical_columns:
        df[column] = df[column].fillna(df[column].mode()[0])

    return df


def encode_categorical_variables(df):
    """Convert categorical variables into numerical values."""

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Separate target variable before encoding
    if "income" in df.columns:
        target = df["income"]
        df = df.drop(columns=["income"])
    else:
        target = None

    # One-hot encode categorical features
    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=True,
        dtype=int
    )

    # Encode target variable
    if target is not None:
        target = target.str.strip().map({
            "<=50K": 0,
            ">50K": 1
        })

        df["income"] = target

    return df


def standardize_numerical_features(df):
    """Standardize numerical features using StandardScaler."""

    # Do not scale the target variable
    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    if "income" in numerical_columns:
        numerical_columns.remove("income")

    scaler = StandardScaler()

    df[numerical_columns] = scaler.fit_transform(
        df[numerical_columns]
    )

    return df


def remove_outliers(df):
    """Remove outliers using the IQR method."""

    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    # Do not use target variable for outlier detection
    if "income" in numerical_columns:
        numerical_columns.remove("income")

    Q1 = df[numerical_columns].quantile(0.25)
    Q3 = df[numerical_columns].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    condition = ~(
        (df[numerical_columns] < lower_bound)
        | (df[numerical_columns] > upper_bound)
    ).any(axis=1)

    df = df[condition]

    return df


def preprocess_data(file_path):
    """Run the complete preprocessing pipeline."""

    # Load dataset
    df = load_data(file_path)

    # Clean column names
    df = clean_column_names(df)

    # Handle missing values
    df = handle_missing_values(df)

    # Encode categorical variables
    df = encode_categorical_variables(df)

    # Standardize numerical features
    df = standardize_numerical_features(df)

    # Remove outliers
    df = remove_outliers(df)

    return df


if __name__ == "__main__":

    input_file = "D:\\Day-1-task-Elevate-Lab---Nishita-kanojiya\\Dataset 2 - Adult\\dataset\\adult.csv"

    output_file = "D:\\Day-1-task-Elevate-Lab---Nishita-kanojiya\\Dataset 2 - Adult\\outputs\\cleaned_adult.csv"

    cleaned_data = preprocess_data(input_file)

    cleaned_data.to_csv(
        output_file,
        index=False
    )

    print("Data preprocessing completed successfully.")
    print(f"Final dataset shape: {cleaned_data.shape}")
    print(f"Cleaned dataset saved to: {output_file}")