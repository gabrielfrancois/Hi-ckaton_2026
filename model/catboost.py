import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Helper for colored printing
try:
    from preprocessing.helper_functions.print import orange, green, red, bold
except ImportError:
    def orange(x): return x
    def green(x): return x
    def red(x): return x
    def bold(x): return x

class PisaCatBoost:
    def __init__(self, random_state=42):
        """
        Initializes the CatBoost Wrapper.
        """
        self.model = None
        self.random_state = random_state
        
        # Hyperparameters for CatBoost
        # Adapted for regression on tabular data (PISA)
        self.params = {
            'iterations': 4000,           # Max trees
            'learning_rate': 0.03,        # Slower learning for better accuracy
            'depth': 8,                   # Standard depth for tabular data
            'l2_leaf_reg': 3,             # Regularization to prevent overfitting
            'loss_function': 'RMSE',      # Regression objective
            'eval_metric': 'RMSE',        # Metric to optimize
            'random_seed': self.random_state,
            'od_type': 'Iter',            # Overfitting Detector type
            'od_wait': 100,               # Early stopping rounds
            'verbose': 200,               # Print every 200 iterations
            'allow_writing_files': False, # Don't spam directories with logs
            'task_type': 'CPU'            # Use 'GPU' if available
        }

    def load_data(self, X_path, y_path=None):
        """
        Loads data using Pandas. 
        """
        print(orange(f"Loading data from {X_path}..."))
        X = pd.read_csv(X_path)
        ids = None
        
        # 1. Capture ID before dropping
        if "CNTSTUID" in X.columns:
            ids = X["CNTSTUID"]
        elif "student_id" in X.columns:
            ids = X["student_id"]
            
        # 2. Drop Non-Feature columns
        cols_to_drop = ["student_id", "Unnamed: 0", "CNTSTUID", "CNTRYID", "CNTSCHID"]
        X = X.drop(columns=cols_to_drop, errors='ignore')
        
        # 3. Load Target (y)
        y = None
        if y_path:
            print(f"Loading target from {y_path}...")
            y_df = pd.read_csv(y_path)
            
            # Identify the correct column
            target_col = None
            if "MathScore" in y_df.columns:
                target_col = "MathScore"
            else:
                target_col = y_df.columns[-1]
            
            print(f"Selected Target Column: {target_col}")
            
            # --- FIX: Use Pandas Selection (Bracket notation) ---
            y = y_df[target_col]
            
            # --- DIAGNOSTICS ---
            print(bold(f"Target Y Stats (Raw) -- Min: {y.min():.2f}, Max: {y.max():.2f}, Mean: {y.mean():.2f}"))

            # --- FILTER OUT ZERO SCORES ---
            # Keep indices aligned by filtering X and y together
            valid_mask = y > 10.0
            
            if not valid_mask.all():
                initial_count = len(y)
                # Filter y
                y = y[valid_mask]
                # Filter X using the index of the valid y rows
                X = X.loc[y.index]
                
                print(red(f"Dropped {initial_count - len(y)} rows where Target Score was <= 10.0"))
                print(bold(f"New Mean Target: {y.mean():.2f}"))

        return X, y, ids

    def align_features(self, X_train, X_test):
        """
        Ensures X_test has exactly the same columns as X_train.
        """
        print(orange("\n--- Aligning Test Features with Training Features ---"))
        train_cols = X_train.columns.tolist()
        test_cols = X_test.columns.tolist()
        
        # 1. Add missing columns (fill with -1)
        missing_in_test = list(set(train_cols) - set(test_cols))
        if missing_in_test:
            print(red(f"Warning: X_test is missing {len(missing_in_test)} columns. Filling with -1."))
            for c in missing_in_test:
                X_test[c] = -1
            
        # 2. Drop extra columns
        extra_in_test = list(set(test_cols) - set(train_cols))
        if extra_in_test:
            print(orange(f"Dropping {len(extra_in_test)} extra columns from X_test."))

        # 3. Reorder columns to match training EXACTLY
        X_test = X_test[train_cols]
        return X_test

    def train(self, X_train, y_train, val_size=0.1):
        """
        Trains the CatBoost model.
        """
        print(orange(f"\n--- Starting CatBoost Training (Split: {1-val_size:.0%} Train / {val_size:.0%} Val) ---"))
        
        # Split data
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, 
            test_size=val_size, 
            random_state=self.random_state
        )

        # Initialize CatBoost Pool (Optimized data structure)
        # CatBoost handles categorical features automatically if specified, 
        # but since we processed them numerically, we pass them as is.
        train_pool = Pool(X_tr, y_tr)
        val_pool = Pool(X_val, y_val)

        self.model = CatBoostRegressor(**self.params)
        
        # Train
        self.model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True
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
        predictions = self.model.predict(X_test)
        
        # Sanity check on predictions
        p_min, p_max, p_mean = predictions.min(), predictions.max(), predictions.mean()
        print(bold(f"Prediction Stats -> Min: {p_min:.2f}, Max: {p_max:.2f}, Mean: {p_mean:.2f}"))
        
        return predictions

    def plot_feature_importance(self, top_n=20):
        print(orange("\n--- Plotting Feature Importance ---"))
        if self.model is None: return

        importance = self.model.get_feature_importance()
        feature_names = self.model.feature_names_
        
        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)
        
        top_features = imp_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(
            x="Importance", y="Feature", data=top_features, 
            palette="magma", hue="Feature", legend=False
        )
        plt.title(f"CatBoost - Top {top_n} Feature Importance")
        plt.xlabel("Importance")
        plt.tight_layout()
        
        if not os.path.exists("model/output"): os.makedirs("model/output")
        plt.savefig("model/output/catboost_importance.png")
        print(green("Feature importance plot saved to 'model/output/catboost_importance.png'"))

    def save_model(self, path="model/output/catboost_pisa_model.cbm"):
        if not os.path.exists("model/output"): os.makedirs("model/output")
        self.model.save_model(path)
        print(green(f"Model saved successfully to {path}"))

if __name__ == "__main__":
    # --- CONFIGURATION ---
    DATA_DIR = "data" 
    
    # Update to your specific timestamps
    X_TRAIN_PATH = f"{DATA_DIR}/X_train_clean_no_leakage_20251130_152757.csv" 
    Y_TRAIN_PATH = f"{DATA_DIR}/y_train.csv"
    X_TEST_PATH = f"{DATA_DIR}/X_test_clean_no_leakage_20251130_152757.csv"   

    cat_pipeline = PisaCatBoost()

    try:
        # 1. LOAD
        X_train, y_train, _ = cat_pipeline.load_data(X_TRAIN_PATH, Y_TRAIN_PATH)
        X_test, _, test_ids = cat_pipeline.load_data(X_TEST_PATH, y_path=None)
        
        # 2. ALIGN
        X_test = cat_pipeline.align_features(X_train, X_test)
        
        # 3. TRAIN
        X_val, y_val = cat_pipeline.train(X_train, y_train, val_size=0.1)
        
        # 4. METRICS
        cat_pipeline.print_validation_metrics(X_val, y_val)
        cat_pipeline.save_model()

        # 5. PREDICT
        test_preds = cat_pipeline.predict(X_test)
        
        # 6. SAVE SUBMISSION
        if test_ids is not None:
            submission_df = pd.DataFrame({
                "CNTSTUID": test_ids,
                "MathScore": test_preds
            })
        else:
            print(red("Warning: No IDs found in test set. Saving predictions with index only."))
            submission_df = pd.DataFrame({"MathScore": test_preds})
            
        submission_path = "model/output/submission_catboost.csv"
        if not os.path.exists("model/output"): os.makedirs("model/output")
        submission_df.to_csv(submission_path, index=False)
        print(green(f"Predictions saved to {submission_path}"))

        # 7. INTERPRET
        cat_pipeline.plot_feature_importance(top_n=20)
        
    except Exception as e:
        print(red(f"Error during execution: {e}"))
        import traceback
        traceback.print_exc()