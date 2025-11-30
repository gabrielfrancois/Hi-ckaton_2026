import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from datetime import datetime

try:
    from preprocessing.helper_functions.print import orange, green, red, bold
except ImportError:
    def orange(x): return x
    def green(x): return x
    def red(x): return x
    def bold(x): return x

class PisaXGBoost:
    def __init__(self, objective='reg:squarederror', random_state=42):
        """
        Initializes the XGBoost Wrapper.
        """
        self.model = None
        self.random_state = random_state
        self.objective = objective
        self.target_is_normalized = False 
        
        # Hyperparameters
        self.params = {
            'n_estimators': 4000,         
            'learning_rate': 0.02,        
            'max_depth': 6,               
            'min_child_weight': 5,        
            'subsample': 0.85,            
            'colsample_bytree': 0.85,     
            'gamma': 0.5,                 
            'reg_alpha': 0.5,             
            'reg_lambda': 1.5,            
            'n_jobs': -1,                 
            'random_state': self.random_state,
            'objective': self.objective,
            'tree_method': 'hist',
            'early_stopping_rounds': 100   
        }

    def load_data(self, X_path, y_path=None):
        """
        Loads data using Pandas. 
        """
        print(orange(f"Loading data from {X_path}..."))
        X = pd.read_csv(X_path)
        ids = None
        
        # 1. Capture ID before dropping (Priority: CNTSTUID -> student_id)
        if "CNTSTUID" in X.columns:
            ids = X["CNTSTUID"]
            
        # 2. Drop Non-Feature columns
        cols_to_drop = ["student_id", "Unnamed: 0", "CNTSTUID", "CNTRYID", "CNTSCHID"]
        # In Pandas, we pass existing columns to drop, errors='ignore' handles the rest
        X = X.drop(columns=cols_to_drop, errors='ignore')
        
        # 3. Load Target (y)
        y = None
        if y_path:
            print(f"Loading target from {y_path}...")
            y_df = pd.read_csv(y_path)
            
            # Cleaning Target: Remove IDs/Index if present
            y_df = y_df.drop(columns=cols_to_drop, errors='ignore')

            # Identify the correct column
            target_col = None
            if "MathScore" in y_df.columns:
                target_col = "MathScore"
            else:
                # Fallback: take the last column if MathScore isn't named explicitly
                target_col = y_df.columns[-1]
            
            print(f"Selected Target Column: {target_col}")
            y = y_df[target_col] # Select as Series
            
            # --- CRITICAL CHECK ---
            y_min = y.min()
            y_max = y.max()
            y_mean = y.mean()
            print(bold(f"Target Y Stats (Raw Load) -- Min: {y_min:.4f}, Max: {y_max:.4f}, Mean: {y_mean:.4f}"))
            
            # Debug: Print head to confirm it's not indices
            print(f"Target Head:\n{y.head()}")

            # --- FILTER OUT ZERO SCORES ---
            # 0.0 is likely missing data.
            valid_mask = y > 0.1
            initial_count = len(y)
            
            # Filter both X and y to maintain alignment
            # Reset index to ensure they match perfectly after filtering
            if not valid_mask.all():
                y = y[valid_mask]
                # We must filter X using the same indices. 
                # Assuming X and y were row-aligned initially.
                X = X.loc[valid_mask.index[valid_mask]] 
                
                dropped_count = initial_count - len(y)
                print(red(f"Dropped {dropped_count} rows where Target Score was <= 0.1."))
                print(f"Training Data Reduced: {initial_count} -> {len(y)}")

            # --- CHECK FOR NORMALIZATION ---
            # Re-calc stats after filtering
            y_mean = y.mean()
            y_max = y.max()
            
            if abs(y_mean) < 10 and y_max < 100:
                print(red("WARNING: Target values detected as Normalized! Predictions will be Denormalized."))
                self.target_is_normalized = True

        return X, y, ids

    def align_features(self, X_train, X_test):
        """
        Ensures X_test has exactly the same columns as X_train.
        """
        print(orange("\n--- Aligning Test Features with Training Features ---"))
        train_cols = X_train.columns.tolist()
        test_cols = X_test.columns.tolist()
        
        # 1. Add missing columns (fill with -1)
        missing_in_test = [c for c in train_cols if c not in test_cols]
        if missing_in_test:
            print(red(f"Warning: X_test is missing {len(missing_in_test)} columns. Filling with -1."))
            # Pandas efficient assignment
            for c in missing_in_test:
                X_test[c] = -1
            
        # 2. Identify extra columns (just for logging)
        extra_in_test = [c for c in test_cols if c not in train_cols]
        if extra_in_test:
            print(orange(f"Dropping {len(extra_in_test)} extra columns from X_test that were not in X_train."))

        # 3. Select exactly the train columns (Enforces order and drops extras)
        X_test = X_test[train_cols]
        return X_test

    def train(self, X_train, y_train, val_size=0.1):
        """
        Trains the XGBoost model using an internal Validation Split.
        """
        print(orange(f"\n--- Starting XGBoost Training (Split: {1-val_size:.0%} Train / {val_size:.0%} Val) ---"))
        
        # No .to_pandas() needed, inputs are already Pandas
        
        # Split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, 
            test_size=val_size, 
            random_state=self.random_state
        )

        self.model = xgb.XGBRegressor(**self.params)
        
        self.model.fit(
            X_tr, y_tr,
            eval_set=[(X_tr, y_tr), (X_val, y_val)],
            verbose=100 
        )
        
        print(green("Training Complete."))
        return X_val, y_val

    def print_validation_metrics(self, X_val, y_val):
        print(orange("\n--- Final Validation Metrics ---"))
        preds = self.model.predict(X_val)
        
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)
        
        print(bold(f"Validation RMSE: {rmse:,.4f}"))
        print(f"Validation MAE:  {mae:,.4f}")
        print(f"Validation R2:   {r2:.4f}")

    def predict(self, X_test):
        print(orange("\n--- Generating Predictions ---"))
        # X_test is already Pandas DataFrame
        predictions = self.model.predict(X_test)
        
        if self.target_is_normalized:
            print(orange("Auto-correcting predictions (Denormalizing)..."))
            predictions = (predictions * 100) + 500
            predictions = np.clip(predictions, 0, 1000)
            
        return predictions

    def plot_feature_importance(self, top_n=20):
        print(orange("\n--- Plotting Feature Importance ---"))
        if self.model is None:
            print(red("Model not trained yet."))
            return

        importance = self.model.feature_importances_
        feature_names = self.model.feature_names_in_
        
        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)
        
        top_features = imp_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        
        sns.barplot(
            x="Importance", 
            y="Feature", 
            data=top_features, 
            palette="viridis",
            hue="Feature",
            legend=False
        )
        
        plt.title(f"XGBoost - Top {top_n} Feature Importance")
        plt.xlabel("Importance (Gain)")
        plt.tight_layout()
        
        if not os.path.exists("model/output"): os.makedirs("model/output")
        plt.savefig("model/output/feature_importance.png")
        print(green("Feature importance plot saved to 'model/output/feature_importance.png'"))

    def save_model(self, path="model/output/xgb_pisa_model.json"):
        if not os.path.exists("model/output"): os.makedirs("model/output")
        self.model.save_model(path)
        print(green(f"Model saved successfully to {path}"))

if __name__ == "__main__":
    DATA_DIR = "data" 
    
    # UPDATE THESE WITH YOUR EXACT FILES
    X_TRAIN_PATH = f"{DATA_DIR}/X_train_preprocessed_20251129_234423.csv" 
    Y_TRAIN_PATH = f"{DATA_DIR}/y_train.csv"
    X_TEST_PATH = f"{DATA_DIR}/X_test_preprocessed_20251129_234429.csv"   

    xgb_pipeline = PisaXGBoost()

    try:
        # 1. LOAD
        X_train, y_train, _ = xgb_pipeline.load_data(X_TRAIN_PATH, Y_TRAIN_PATH)
        X_test, _, test_ids = xgb_pipeline.load_data(X_TEST_PATH, y_path=None)
        
        # 2. ALIGN
        X_test = xgb_pipeline.align_features(X_train, X_test)
        
        # 3. TRAIN
        X_val, y_val = xgb_pipeline.train(X_train, y_train, val_size=0.1)
        
        # 4. SAVE MODEL
        xgb_pipeline.save_model()
        
        # 5. METRICS
        xgb_pipeline.print_validation_metrics(X_val, y_val)

        # 6. PREDICT
        test_preds = xgb_pipeline.predict(X_test)
        
        # 7. SAVE SUBMISSION
        if test_ids is not None:
            submission_df = pd.DataFrame({
                "ID": test_ids,
                "MathScore": test_preds
            })
        else:
            print(red("Warning: No IDs found in test set. Saving predictions with index only."))
            submission_df = pd.DataFrame({"MathScore": test_preds})
            
        submission_path = "model/output/submission.csv"
        if not os.path.exists("model/output"): os.makedirs("model/output")
        
        submission_df.to_csv(submission_path, index=False)
        print(green(f"Predictions saved to {submission_path} (Shape: {submission_df.shape})"))

        # 8. INTERPRET
        xgb_pipeline.plot_feature_importance(top_n=20)
        
    except Exception as e:
        print(red(f"Error during execution: {e}"))
        print("Please check filenames and ensure data exists.")