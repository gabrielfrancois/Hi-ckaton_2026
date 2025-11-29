import polars as pl
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

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
        
        # Hyperparameters
        self.params = {
            'n_estimators': 2000,         
            'learning_rate': 0.02,        
            'max_depth': 8,               
            'subsample': 0.85,            
            'colsample_bytree': 0.85,     
            'gamma': 0.1,                 
            'reg_alpha': 0.5,             
            'reg_lambda': 1.5,            
            'n_jobs': -1,                 
            'random_state': self.random_state,
            'objective': self.objective,
            'tree_method': 'hist',
            'early_stopping_rounds': 50   
        }

    def load_data(self, X_path, y_path=None):
        """
        Loads data using Polars. y_path is optional for the test set.
        """
        print(orange(f"Loading data from {X_path}..."))
        X = pl.read_csv(X_path)
        
        # Drop ID/Index columns if they exist
        cols_to_drop = ["student_id", "Unnamed: 0"]
        existing_drop = [c for c in cols_to_drop if c in X.columns]
        
        if existing_drop:
            print(f"Dropping {existing_drop} from features...")
            X = X.drop(existing_drop)
        
        y = None
        if y_path:
            print(f"Loading target from {y_path}...")
            y = pl.read_csv(y_path)
            # Drop ID columns from target too if present
            existing_drop_y = [c for c in cols_to_drop if c in y.columns]
            if existing_drop_y:
                y = y.drop(existing_drop_y)
            
            # --- CRITICAL FIX: Ensure Target is 1D ---
            # If y still has multiple columns (e.g. index not caught), select the last one.
            if y.width > 1:
                print(red(f"Warning: Target 'y' has {y.width} columns: {y.columns}. Auto-selecting the last column as target."))
                y = y.select(y.columns[-1])

        return X, y

    def align_features(self, X_train, X_test):
        """
        Ensures X_test has exactly the same columns as X_train, in the same order.
        """
        print(orange("\n--- Aligning Test Features with Training Features ---"))
        train_cols = X_train.columns
        test_cols = X_test.columns
        
        # 1. Identify columns missing in Test that are in Train
        missing_in_test = [c for c in train_cols if c not in test_cols]
        if missing_in_test:
            print(red(f"Warning: X_test is missing {len(missing_in_test)} columns used in training. Filling with -1."))
            X_test = X_test.with_columns([
                pl.lit(-1).alias(c) for c in missing_in_test
            ])
            
        # 2. Identify columns in Test that are NOT in Train (Extras)
        extra_in_test = [c for c in test_cols if c not in train_cols]
        if extra_in_test:
            print(orange(f"Dropping {len(extra_in_test)} extra columns from X_test that were not in X_train."))

        # 3. Select exactly the train columns
        X_test = X_test.select(train_cols)
        
        return X_test

    def train(self, X_train, y_train, val_size=0.1):
        """
        Trains the XGBoost model using an internal Validation Split.
        Returns the validation data so we can print metrics later.
        """
        print(orange(f"\n--- Starting XGBoost Training (Split: {1-val_size:.0%} Train / {val_size:.0%} Val) ---"))
        
        # 1. Convert to Pandas
        X_pd = X_train.to_pandas()
        y_pd = y_train.to_pandas()

        # 2. Create Internal Validation Set
        # Ensure y is flattened to 1D array to avoid array[f32, 2] output
        y_pd = y_pd.iloc[:, 0] if len(y_pd.shape) > 1 and y_pd.shape[1] == 1 else y_pd

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_pd, y_pd, 
            test_size=val_size, 
            random_state=self.random_state
        )

        # 3. Initialize Model
        self.model = xgb.XGBRegressor(**self.params)
        
        # 4. Fit model
        self.model.fit(
            X_tr, y_tr,
            eval_set=[(X_tr, y_tr), (X_val, y_val)],
            verbose=100 
        )
        
        print(green("Training Complete."))
        
        # Return validation set for metric printing
        return X_val, y_val

    def print_validation_metrics(self, X_val, y_val):
        """
        Explicitly prints metrics for the validation set.
        """
        print(orange("\n--- Final Validation Metrics ---"))
        preds = self.model.predict(X_val)
        
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)
        
        print(bold(f"Validation RMSE: {rmse:,.4f}"))
        print(f"Validation MAE:  {mae:,.4f}")
        print(f"Validation R2:   {r2:.4f}")

    def predict(self, X_test):
        """
        Generates predictions for a dataset.
        """
        print(orange("\n--- Generating Predictions ---"))
        X_test_np = X_test.to_pandas()
        predictions = self.model.predict(X_test_np)
        return predictions

    def plot_feature_importance(self, top_n=20):
        print(orange("\n--- Plotting Feature Importance ---"))
        importance = self.model.feature_importances_
        feature_names = self.model.feature_names_in_
        
        imp_df = pl.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort("Importance", descending=True)
        
        top_features = imp_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x="Importance", y="Feature", data=top_features.to_pandas(), palette="viridis")
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
    # --- CONFIGURATION ---
    DATA_DIR = "data" 
    
    X_TRAIN_PATH = f"{DATA_DIR}/X_train_preprocessed_20251129_210719.csv" 
    Y_TRAIN_PATH = f"{DATA_DIR}/y_train.csv"
    X_TEST_PATH = f"{DATA_DIR}/X_test_preprocessed_20251129_210728.csv"   

    # --- 1. INSTANTIATE ---
    xgb_pipeline = PisaXGBoost()

    # --- 2. LOAD DATA ---
    try:
        X_train, y_train = xgb_pipeline.load_data(X_TRAIN_PATH, Y_TRAIN_PATH)
        X_test, _ = xgb_pipeline.load_data(X_TEST_PATH, y_path=None)
        
        # --- 3. ALIGN DATA ---
        X_test = xgb_pipeline.align_features(X_train, X_test)
        
        # --- 4. TRAIN (Split & Return Validation Data) ---
        X_val, y_val = xgb_pipeline.train(X_train, y_train, val_size=0.1)
        
        # --- 5. SAFETY SAVE MODEL (Before prediction crashes) ---
        xgb_pipeline.save_model()
        
        # --- 6. DISPLAY METRICS ---
        xgb_pipeline.print_validation_metrics(X_val, y_val)

        # --- 7. PREDICT ON X_TEST ---
        test_preds = xgb_pipeline.predict(X_test)
        
        # Flatten predictions just in case
        if len(test_preds.shape) > 1:
            test_preds = test_preds.flatten()

        # Save Predictions
        submission_df = pl.DataFrame({"predicted_score": test_preds})
        submission_path = "model/output/submission.csv"
        if not os.path.exists("model/output"): os.makedirs("model/output")
        submission_df.write_csv(submission_path)
        print(green(f"Predictions saved to {submission_path}"))

        # --- 8. INTERPRET ---
        xgb_pipeline.plot_feature_importance(top_n=20)
        
    except Exception as e:
        print(red(f"Error during execution: {e}"))
        print("Please check filenames and ensure data exists.")