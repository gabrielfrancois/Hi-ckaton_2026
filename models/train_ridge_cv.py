"""
Train Ridge Regression with Cross-Validation using ALL features.

This script uses nested cross-validation:
- Outer CV (5-fold): To evaluate model performance
- Inner CV: RidgeCV automatically finds optimal alpha for each outer fold
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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
        y_train = y_train['MathScore']
    elif y_train.shape[1] > 1:
        print(f"Warning: y_train has {y_train.shape[1]} columns. Using the last column.")
        y_train = y_train.iloc[:, -1]

    return X_train, y_train

def main():
    """Main execution function."""
    print("Ridge Regression with Nested Cross-Validation")
    print("Using ALL features\n")

    # Load data
    X, y = load_latest_preprocessed_data()
    print(f"\nDataset shape: {X.shape}")
    print(f"Total features: {X.shape[1]}")
    print(f"Total samples: {X.shape[0]}")

    # Create pipeline with standardization and Ridge (with internal CV for alpha)
    print("\nCreating pipeline: StandardScaler + RidgeCV")
    alphas = np.logspace(-2, 6, 50)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RidgeCV(alphas=alphas, cv=5))
    ])

    # Perform outer cross-validation
    print("\nPerforming 5-fold Cross-Validation (outer loop)...")
    print("Note: RidgeCV performs inner CV for alpha selection in each fold")
    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=5,
        scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'],
        return_train_score=True,
        n_jobs=-1,
        verbose=1
    )

    print("✓ Cross-validation complete")

    # Calculate statistics
    train_r2_mean = cv_results['train_r2'].mean()
    train_r2_std = cv_results['train_r2'].std()
    test_r2_mean = cv_results['test_r2'].mean()
    test_r2_std = cv_results['test_r2'].std()

    train_mse_mean = -cv_results['train_neg_mean_squared_error'].mean()
    train_mse_std = cv_results['train_neg_mean_squared_error'].std()
    test_mse_mean = -cv_results['test_neg_mean_squared_error'].mean()
    test_mse_std = cv_results['test_neg_mean_squared_error'].std()

    train_mae_mean = -cv_results['train_neg_mean_absolute_error'].mean()
    train_mae_std = cv_results['train_neg_mean_absolute_error'].std()
    test_mae_mean = -cv_results['test_neg_mean_absolute_error'].mean()
    test_mae_std = cv_results['test_neg_mean_absolute_error'].std()

    # Print results
    print("\n" + "="*70)
    print("Cross-Validation Results (5-fold)")
    print("="*70)

    print("\nTrain Set Performance:")
    print(f"  R² Score:  {train_r2_mean:.6f} ± {train_r2_std:.6f}")
    print(f"  RMSE:      {np.sqrt(train_mse_mean):.4f} ± {np.sqrt(train_mse_std):.4f}")
    print(f"  MAE:       {train_mae_mean:.4f} ± {train_mae_std:.4f}")

    print("\nTest Set Performance:")
    print(f"  R² Score:  {test_r2_mean:.6f} ± {test_r2_std:.6f}")
    print(f"  RMSE:      {np.sqrt(test_mse_mean):.4f} ± {np.sqrt(test_mse_std):.4f}")
    print(f"  MAE:       {test_mae_mean:.4f} ± {test_mae_std:.4f}")

    print("\nOverfitting Analysis:")
    print(f"  R² difference: {abs(train_r2_mean - test_r2_mean):.6f}")
    if abs(train_r2_mean - test_r2_mean) < 0.05:
        print("  ✓ Good generalization (low overfitting)")
    elif abs(train_r2_mean - test_r2_mean) < 0.10:
        print("  ⚠ Moderate overfitting")
    else:
        print("  ⚠ Significant overfitting detected")

    # Individual fold results
    print("\n" + "="*70)
    print("Individual Fold Results (Test R²)")
    print("="*70)
    for i, r2 in enumerate(cv_results['test_r2'], 1):
        print(f"  Fold {i}: {r2:.6f}")

    # Train final model on all data
    print("\n" + "="*70)
    print("Training final model on all data...")
    print("="*70)
    pipeline.fit(X, y)
    print("✓ Final model trained")
    print(f"Optimal alpha found: {pipeline.named_steps['model'].alpha_:.4f}")

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    model_filename = os.path.join(save_dir, f"ridge_cv_all_features_{timestamp}.pkl")

    import pickle
    with open(model_filename, 'wb') as f:
        pickle.dump(pipeline, f)

    print(f"\n✓ Model saved to: {model_filename}")

    # Save CV results
    results_filename = os.path.join(save_dir, f"cv_results_ridge_{timestamp}.csv")
    results_df = pd.DataFrame({
        'Fold': range(1, 6),
        'Train_R2': cv_results['train_r2'],
        'Test_R2': cv_results['test_r2'],
        'Train_MSE': -cv_results['train_neg_mean_squared_error'],
        'Test_MSE': -cv_results['test_neg_mean_squared_error'],
        'Train_MAE': -cv_results['train_neg_mean_absolute_error'],
        'Test_MAE': -cv_results['test_neg_mean_absolute_error']
    })
    results_df.to_csv(results_filename, index=False)
    print(f"✓ CV results saved to: {results_filename}")

    print("\nCross-validation complete!")

if __name__ == "__main__":
    main()
