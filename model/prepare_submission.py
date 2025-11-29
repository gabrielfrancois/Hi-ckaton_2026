"""
Prepare submission file for Hickathon competition.

This script loads a trained model, makes predictions on the test set,
and creates a submission file in the required format.
"""

import pandas as pd
import numpy as np
import pickle
import os
import glob
from datetime import datetime


def load_latest_test_data():
    """Load the most recent preprocessed test data and extract IDs from original X_test.csv."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    # Find the latest preprocessed test file
    x_test_pattern = os.path.join(data_dir, 'X_test_preprocessed_*.csv')
    x_test_files = glob.glob(x_test_pattern)

    if not x_test_files:
        raise FileNotFoundError(f"No preprocessed X_test file found in {data_dir}")

    latest_x_test = max(x_test_files, key=os.path.getctime)

    print(f"Loading X_test from: {latest_x_test}")
    X_test = pd.read_csv(latest_x_test)

    # Remove ID column if present in preprocessed file (shouldn't be there)
    if 'Unnamed: 0' in X_test.columns:
        X_test = X_test.drop('Unnamed: 0', axis=1)
    elif 'ID' in X_test.columns:
        X_test = X_test.drop('ID', axis=1)

    # Load IDs from the original X_test.csv file
    original_test_path = os.path.join(data_dir, 'X_test.csv')
    if not os.path.exists(original_test_path):
        raise FileNotFoundError(f"Original test file not found: {original_test_path}")

    print(f"Loading IDs from: {original_test_path}")
    # Only load the ID column to save memory
    test_ids = pd.read_csv(original_test_path, usecols=['Unnamed: 0'])['Unnamed: 0'].astype(int).values

    # Verify that the number of IDs matches the number of test samples
    if len(test_ids) != len(X_test):
        raise ValueError(
            f"Mismatch between number of IDs ({len(test_ids)}) "
            f"and test samples ({len(X_test)}). "
            f"The order of rows may have been modified during preprocessing!"
        )

    print(f"✓ Loaded {len(test_ids)} test samples with IDs from original file")

    return X_test, test_ids


def load_train_columns():
    """Load the column names from the training data to align test data."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    # Find the latest preprocessed train file
    x_train_pattern = os.path.join(data_dir, 'X_train_preprocessed_*.csv')
    x_train_files = glob.glob(x_train_pattern)

    if not x_train_files:
        raise FileNotFoundError(f"No preprocessed X_train file found in {data_dir}")

    latest_x_train = max(x_train_files, key=os.path.getctime)

    print(f"Loading column names from: {latest_x_train}")
    # Only read the first row to get column names
    train_columns = pd.read_csv(latest_x_train, nrows=0).columns.tolist()

    # Remove ID column if present
    train_columns = [col for col in train_columns if col not in ['Unnamed: 0', 'ID']]

    return train_columns


def align_test_data_with_train(X_test, train_columns):
    """Align test data columns with training data columns."""
    print(f"\nAligning test data columns with training data...")
    print(f"Test data has {len(X_test.columns)} columns")
    print(f"Training data had {len(train_columns)} columns")

    # Find missing columns
    missing_cols = set(train_columns) - set(X_test.columns)
    extra_cols = set(X_test.columns) - set(train_columns)

    if missing_cols:
        print(f"Adding {len(missing_cols)} missing columns with zeros")
        for col in missing_cols:
            X_test[col] = 0

    if extra_cols:
        print(f"Removing {len(extra_cols)} extra columns")

    # Reorder columns to match training data exactly
    X_test_aligned = X_test[train_columns]

    print(f"✓ Test data aligned: {X_test_aligned.shape}")

    return X_test_aligned


def load_model(model_path):
    """Load a trained model from pickle file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model from: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    return model


def load_scaler(model_path):
    """
    Load the scaler associated with the model.
    Tries to find a scaler file with the same timestamp as the model.
    """
    # Extract the timestamp from the model filename
    model_dir = os.path.dirname(model_path)
    model_filename = os.path.basename(model_path)

    # Try to find matching scaler
    # Pattern: model_name_timestamp.pkl -> scaler_name_timestamp.pkl or name_scaler_timestamp.pkl
    if 'linear_regression' in model_filename:
        scaler_pattern = model_filename.replace('linear_regression', 'scaler')
    elif 'lasso' in model_filename:
        scaler_pattern = model_filename.replace('lasso', 'lasso_scaler')
    elif 'ridge' in model_filename:
        scaler_pattern = model_filename.replace('ridge', 'ridge_scaler')
    else:
        # Generic pattern
        timestamp = model_filename.split('_')[-1].replace('.pkl', '')
        scaler_pattern = f"*scaler*{timestamp}.pkl"

    scaler_path = os.path.join(model_dir, scaler_pattern)
    scaler_files = glob.glob(scaler_path)

    if scaler_files:
        scaler_path = scaler_files[0]
        print(f"Loading scaler from: {scaler_path}")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return scaler
    else:
        print("Warning: No scaler found. Predictions will be made without scaling.")
        return None


def create_submission(model_path):
    """Create a submission file using the specified model."""
    # Load test data
    X_test, test_ids = load_latest_test_data()
    print(f"Test set shape: {X_test.shape}")
    print(f"Number of test samples: {len(test_ids)}")

    # Load training columns and align test data
    train_columns = load_train_columns()
    X_test = align_test_data_with_train(X_test, train_columns)

    # Load model
    model = load_model(model_path)

    # Load scaler
    scaler = load_scaler(model_path)

    # Scale features if scaler is available
    if scaler is not None:
        print("\nScaling test features...")
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test

    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X_test_scaled)

    # Flatten predictions if needed
    if predictions.ndim > 1:
        predictions = predictions.flatten()

    # Create submission dataframe
    submission = pd.DataFrame({
        'ID': test_ids,
        'MathScore': predictions
    })

    # Generate output filename
    model_name = os.path.basename(model_path).replace('.pkl', '')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{model_name}_{timestamp}.csv"

    # Create submissions directory if it doesn't exist
    submissions_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'submissions')
    os.makedirs(submissions_dir, exist_ok=True)

    output_path = os.path.join(submissions_dir, output_filename)

    # Save submission file
    submission.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print("Submission File Created Successfully!")
    print(f"{'='*60}")
    print(f"Output file: {output_path}")
    print(f"Number of predictions: {len(submission)}")
    print(f"\nFirst few predictions:")
    print(submission.head(10).to_string(index=False))
    print(f"\nPrediction statistics:")
    print(f"  Mean:   {predictions.mean():.4f}")
    print(f"  Median: {np.median(predictions):.4f}")
    print(f"  Min:    {predictions.min():.4f}")
    print(f"  Max:    {predictions.max():.4f}")
    print(f"  Std:    {predictions.std():.4f}")

    return output_path


def main():
    """Main execution function."""
    print("="*60)
    print("Submission Preparation Script")
    print("="*60)

    # Ask user for model path
    print("\nPlease enter the path to the model (.pkl file):")
    print("Example: models/saved_models/linear_regression_all_features_20251129_214302.pkl")
    model_path = input("\nModel path: ").strip()

    # Handle relative paths
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)

    try:
        output_path = create_submission(model_path)
        print(f"\n✓ Submission ready: {output_path}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
