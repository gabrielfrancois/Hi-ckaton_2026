"""
Train Linear Regression model using top 9 features from feature selection analysis.

Based on Mutual Information scores, the following features are used:
1. math_q2_average_score (MI: 0.897)
2. math_q3_average_score (MI: 0.888)
3. math_q1_average_score (MI: 0.835)
4. math_q4_average_score (MI: 0.798)
5. Year (MI: 0.744)
6. math_q5_average_score (MI: 0.726)
7. math_q6_average_score (MI: 0.672)
8. math_q9_average_score (MI: 0.661)
9. math_q8_average_score (MI: 0.654)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import glob
import os
from datetime import datetime

# Define the 9 top features (excluding the empty column " ")
TOP_FEATURES = [
    'math_q2_average_score',
    'math_q3_average_score',
    'math_q1_average_score',
    'math_q4_average_score',
    'Year',
    'math_q5_average_score',
    'math_q6_average_score',
    'math_q9_average_score',
    'math_q8_average_score'
]

def load_latest_preprocessed_data():
    """Load the most recent preprocessed training data."""
    # Get the script's directory and construct absolute path to data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')

    # Find the latest preprocessed files
    x_train_pattern = os.path.join(data_dir, 'X_train_preprocessed_*.csv')
    x_train_files = glob.glob(x_train_pattern)
    if not x_train_files:
        raise FileNotFoundError(f"No preprocessed X_train file found in {data_dir}")

    latest_x_train = max(x_train_files, key=os.path.getctime)

    # Load y_train
    y_train_path = os.path.join(data_dir, 'y_train.csv')
    if not os.path.exists(y_train_path):
        raise FileNotFoundError(f"y_train file not found at {y_train_path}")

    print(f"Loading X_train from: {latest_x_train}")
    print(f"Loading y_train from: {y_train_path}")

    X_train = pd.read_csv(latest_x_train)
    y_train = pd.read_csv(y_train_path)

    # Keep only MathScore column if y_train has multiple columns
    if 'MathScore' in y_train.columns:
        y_train = y_train[['MathScore']]
    elif y_train.shape[1] > 1:
        print(f"Warning: y_train has {y_train.shape[1]} columns. Using the last column.")
        y_train = y_train.iloc[:, -1:]

    return X_train, y_train

def select_top_features(X_train):
    """Select only the top 9 features from the dataset."""
    missing_features = [f for f in TOP_FEATURES if f not in X_train.columns]

    if missing_features:
        print(f"Warning: Missing features in dataset: {missing_features}")
        available_features = [f for f in TOP_FEATURES if f in X_train.columns]
        print(f"Using {len(available_features)} available features")
        return X_train[available_features]

    return X_train[TOP_FEATURES]

def train_linear_regression(X_train, y_train):
    """Train a linear regression model."""
    print("\n" + "="*60)
    print("Training Linear Regression Model")
    print("="*60)

    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"\nFeatures used:")
    for i, feature in enumerate(X_train.columns, 1):
        print(f"  {i}. {feature}")

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on training data
    y_pred = model.predict(X_train)

    # Calculate metrics
    mse = mean_squared_error(y_train, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)

    # Print results
    print("\n" + "="*60)
    print("Training Results")
    print("="*60)
    print(f"R² Score:  {r2:.6f}")
    print(f"RMSE:      {rmse:.4f}")
    print(f"MAE:       {mae:.4f}")
    print(f"MSE:       {mse:.4f}")

    # Print feature coefficients
    print("\n" + "="*60)
    print("Feature Coefficients")
    print("="*60)

    # Handle both 1D and 2D coefficient arrays
    coefficients = model.coef_.flatten()

    coef_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': coefficients
    }).sort_values('Coefficient', key=abs, ascending=False)

    print(coef_df.to_string(index=False))

    # Handle both scalar and array intercepts
    if isinstance(model.intercept_, np.ndarray):
        intercept = model.intercept_[0] if len(model.intercept_) > 0 else model.intercept_
    else:
        intercept = model.intercept_
    print(f"\nIntercept: {intercept:.4f}")

    return model

def main():
    """Main execution function."""
    print("Linear Regression Training Script")
    print("Using top 9 features from Mutual Information analysis\n")

    # Load data
    X_train, y_train = load_latest_preprocessed_data()
    print(f"\nOriginal X_train shape: {X_train.shape}")

    # Select top features
    X_train_selected = select_top_features(X_train)
    print(f"Selected features shape: {X_train_selected.shape}")

    # Train model
    model = train_linear_regression(X_train_selected, y_train)

    # Save model (optional)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    model_filename = os.path.join(save_dir, f"linear_regression_top9_{timestamp}.pkl")

    import pickle
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)

    print(f"\n✓ Model saved to: {model_filename}")
    print("\nTraining complete!")

if __name__ == "__main__":
    main()
