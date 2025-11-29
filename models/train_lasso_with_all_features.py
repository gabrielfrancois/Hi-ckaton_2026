"""
Train Lasso Regression model with train/test split using ALL features.

This script trains a Lasso regression model (L1 regularization) using all available features
from the preprocessed dataset. Lasso can help with feature selection by forcing some
coefficients to exactly zero.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import glob
import os
from datetime import datetime

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

def train_and_evaluate(X_train, X_test, y_train, y_test, use_cv=True):
    """Train a Lasso regression model and evaluate on both train and test sets."""
    print("\n" + "="*60)
    print("Training Lasso Regression Model")
    print("="*60)

    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    print(f"Number of features: {X_train.shape[1]}")

    # Standardize features for better Lasso performance
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✓ Standardization complete")

    # Train the model
    if use_cv:
        print("\nUsing LassoCV to find optimal alpha...")
        # Test different alpha values
        alphas = np.logspace(-2, 6, 50)
        model = LassoCV(alphas=alphas, cv=5, max_iter=50000, random_state=42, n_jobs=-1)
        model.fit(X_train_scaled, y_train.values.ravel())
        print(f"✓ Optimal alpha found: {model.alpha_:.4f}")
    else:
        print("\nTraining Lasso with alpha=1.0...")
        model = Lasso(alpha=1.0, max_iter=50000, random_state=42)
        model.fit(X_train_scaled, y_train.values.ravel())

    print("✓ Training complete")

    # Make predictions on both train and test sets (using scaled data)
    print("\nMaking predictions...")
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Calculate metrics for training set
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    # Calculate metrics for test set
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Print results
    print("\n" + "="*60)
    print("Training Set Results")
    print("="*60)
    print(f"R² Score:  {train_r2:.6f}")
    print(f"RMSE:      {train_rmse:.4f}")
    print(f"MAE:       {train_mae:.4f}")
    print(f"MSE:       {train_mse:.4f}")

    print("\n" + "="*60)
    print("Test Set Results")
    print("="*60)
    print(f"R² Score:  {test_r2:.6f}")
    print(f"RMSE:      {test_rmse:.4f}")
    print(f"MAE:       {test_mae:.4f}")
    print(f"MSE:       {test_mse:.4f}")

    # Analyze feature selection
    coefficients = model.coef_.flatten()
    non_zero_coefs = np.sum(coefficients != 0)
    zero_coefs = np.sum(coefficients == 0)

    print("\n" + "="*60)
    print("Feature Selection Analysis")
    print("="*60)
    print(f"Total features: {len(coefficients)}")
    print(f"Non-zero coefficients: {non_zero_coefs}")
    print(f"Zero coefficients: {zero_coefs}")
    print(f"Features eliminated: {zero_coefs / len(coefficients) * 100:.2f}%")

    # Print top 20 non-zero feature coefficients by absolute value
    print("\n" + "="*60)
    print("Top 20 Non-Zero Feature Coefficients (by absolute value)")
    print("="*60)

    coef_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': coefficients
    })

    # Filter non-zero and sort by absolute value
    non_zero_coef_df = coef_df[coef_df['Coefficient'] != 0].copy()
    non_zero_coef_df['Abs_Coefficient'] = non_zero_coef_df['Coefficient'].abs()
    non_zero_coef_df = non_zero_coef_df.sort_values('Abs_Coefficient', ascending=False)

    print(non_zero_coef_df[['Feature', 'Coefficient']].head(20).to_string(index=False))

    print(f"\nIntercept: {model.intercept_:.4f}")

    return model, scaler, train_r2, test_r2, coef_df

def main():
    """Main execution function."""
    print("Lasso Regression Training Script with ALL Features")
    print("Using train/test split (70/30)\n")

    # Load data
    X, y = load_latest_preprocessed_data()
    print(f"\nOriginal dataset shape: {X.shape}")
    print(f"Total features: {X.shape[1]}")

    # Split into train (70%) and test (30%)
    print("\nSplitting data: 70% train, 30% test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42
    )

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Train and evaluate model
    model, scaler, train_r2, test_r2, coef_df = train_and_evaluate(X_train, X_test, y_train, y_test, use_cv=True)

    # Save model and scaler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    model_filename = os.path.join(save_dir, f"lasso_all_features_{timestamp}.pkl")
    scaler_filename = os.path.join(save_dir, f"lasso_scaler_{timestamp}.pkl")

    import pickle
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_filename, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"\n✓ Model saved to: {model_filename}")
    print(f"✓ Scaler saved to: {scaler_filename}")

    # Save coefficients to CSV
    coef_filename = os.path.join(save_dir, f"lasso_coefficients_all_features_{timestamp}.csv")
    coef_df.to_csv(coef_filename, index=False)
    print(f"✓ Coefficients saved to: {coef_filename}")

    # Save selected features (non-zero coefficients)
    selected_features = coef_df[coef_df['Coefficient'] != 0].copy()
    selected_features['Abs_Coefficient'] = selected_features['Coefficient'].abs()
    selected_features = selected_features.sort_values('Abs_Coefficient', ascending=False)

    selected_filename = os.path.join(save_dir, f"lasso_selected_features_{timestamp}.csv")
    selected_features[['Feature', 'Coefficient']].to_csv(selected_filename, index=False)
    print(f"✓ Selected features saved to: {selected_filename}")

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Total features: {X.shape[1]}")
    print(f"Selected features: {len(selected_features)}")
    if hasattr(model, 'alpha_'):
        print(f"Optimal alpha: {model.alpha_:.4f}")
    print(f"Train R²: {train_r2:.6f}")
    print(f"Test R²:  {test_r2:.6f}")
    print(f"Difference: {abs(train_r2 - test_r2):.6f}")

    if abs(train_r2 - test_r2) < 0.05:
        print("✓ Good generalization (low overfitting)")
    elif abs(train_r2 - test_r2) < 0.10:
        print("⚠ Moderate overfitting")
    else:
        print("⚠ Significant overfitting detected")

    print("\nTraining complete!")

if __name__ == "__main__":
    main()
