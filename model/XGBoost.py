import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor
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
    def __init__(self, random_state=42):
        """
        Initializes the XGBoost Wrapper inside a TransformedTargetRegressor.
        """
        self.wrapper_model = None 
        self.random_state = random_state
        
        # Hyperparameters
        self.params = {
            'n_estimators': 4000,         
            'learning_rate': 0.02,        
            'max_depth': 6,               
            'min_child_weight': 10,       
            'subsample': 0.85,            
            'colsample_bytree': 0.8,     
            'gamma': 0.5,                 
            'reg_alpha': 1.0,             
            'reg_lambda': 3.0,            
            'n_jobs': -1,                 
            'random_state': self.random_state,
            'objective': 'reg:squarederror', 
            'tree_method': 'hist',
            'early_stopping_rounds': 100   
        }

    def load_data(self, X_path, y_path=None):
        print(orange(f"Loading data from {X_path}..."))
        X = pd.read_csv(X_path)
        ids = None
        
        # 1. Capture ID
        if "CNTSTUID" in X.columns:
            ids = X["CNTSTUID"]
        elif "student_id" in X.columns:
            ids = X["student_id"]
            
        # 2. Drop Non-Feature columns & LEAKY COLUMNS
        cols_to_drop = ["student_id", "Unnamed: 0", "CNTSTUID", "CNTRYID", "CNTSCHID", "MathScore"]
        X = X.drop(columns=cols_to_drop, errors='ignore')
        
        # --- FIX: DROP NON-NUMERIC COLUMNS (CNT, CYC, STRATUM) ---
        # XGBoost requires all inputs to be numeric.
        initial_cols = X.shape[1]
        X = X.select_dtypes(include=[np.number])
        dropped_cols = initial_cols - X.shape[1]
        if dropped_cols > 0:
            print(orange(f"Dropped {dropped_cols} non-numeric columns (objects/strings)."))
        
        # 3. Load Target (y)
        y = None
        if y_path:
            print(f"Loading target from {y_path}...")
            y_df = pd.read_csv(y_path)
            
            if "MathScore" in y_df.columns:
                target_col = "MathScore"
            else:
                target_col = y_df.columns[-1]
            
            y = y_df[target_col]
            
            # --- FILTER ZERO SCORES ---
            valid_mask = y > 10.0
            if not valid_mask.all():
                initial_count = len(y)
                X = X.loc[valid_mask]
                y = y[valid_mask]
                print(red(f"Dropped {initial_count - len(y)} rows where Target <= 10.0"))

            print(bold(f"Final Target Stats -> Min: {y.min():.2f}, Max: {y.max():.2f}, Mean: {y.mean():.2f}"))

        return X, y, ids

    def align_features(self, X_train, X_test):
        print(orange("\n--- Aligning Test Features with Training Features ---"))
        train_cols = X_train.columns.tolist()
        test_cols = X_test.columns.tolist()
        
        # 1. Drop Non-Numeric in Test first
        X_test = X_test.select_dtypes(include=[np.number])
        
        # 2. Add missing columns
        missing = [c for c in train_cols if c not in X_test.columns]
        if missing:
            print(red(f"Warning: X_test missing {len(missing)} columns. Filling with -1."))
            for c in missing:
                X_test[c] = -1
            
        # 3. Select exact columns
        X_test = X_test[train_cols]
        return X_test

    def train(self, X_train, y_train, val_size=0.1):
        print(orange(f"\n--- Starting XGBoost Training with TransformedTargetRegressor ---"))
        
        self.train_features_ = X_train.columns.tolist()
        
        # 1. Split (on RAW target)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, 
            test_size=val_size, 
            random_state=self.random_state
        )

        # 2. Initialize Inner Regressor
        xgb_inner = xgb.XGBRegressor(**self.params)
        
        # 3. Initialize Wrapper
        # This handles y -> log(1+y) automatically
        self.wrapper_model = TransformedTargetRegressor(
            regressor=xgb_inner,
            func=np.log1p,
            inverse_func=np.expm1
        )
        
        # 4. Fit
        # CRITICAL: We must manually transform y_val for the eval_set because 
        # the wrapper only transforms the training y passed to fit().
        print(orange("Fitting model (Log-Transform applied internally)..."))
        
        y_tr_log = np.log1p(y_tr)
        y_val_log = np.log1p(y_val)
        
        # We pass the eval_set directly. XGBoost will use this for early stopping.
        # Since the inner model sees log-transformed targets, eval_set must also be log-transformed.
        self.wrapper_model.fit(
            X_tr, y_tr,
            eval_set=[(X_tr, y_tr_log), (X_val, y_val_log)],
            verbose=200
        )
        
        print(green("Training Complete."))
        return X_val, y_val

    def print_validation_metrics(self, X_val, y_val):
        print(orange("\n--- Final Validation Metrics ---"))
        
        # Predict uses the wrapper, so it automatically inverses the transform
        # We get Real Scores back immediately
        preds = self.wrapper_model.predict(X_val)
        
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)
        
        print(bold(f"Validation RMSE: {rmse:,.4f}"))
        print(f"Validation MAE:  {mae:,.4f}")
        print(f"Validation R2:   {r2:.4f}")

    def predict(self, X_test):
        print(orange("\n--- Generating Predictions ---"))
        
        # Align columns
        X_test = X_test.reindex(columns=self.train_features_, fill_value=-1)
        
        # Prediction is automatic (Inverse transform included)
        predictions = self.wrapper_model.predict(X_test)
        
        # Final safety clip
        predictions = np.clip(predictions, 0, 1000)
        
        p_min, p_max, p_mean = predictions.min(), predictions.max(), predictions.mean()
        print(bold(f"Prediction Stats -> Min: {p_min:.2f}, Max: {p_max:.2f}, Mean: {p_mean:.2f}"))
        
        return predictions

    def plot_feature_importance(self, top_n=20):
        print(orange("\n--- Plotting Feature Importance ---"))
        if self.wrapper_model is None: return

        # Extract inner model to get importance
        inner_model = self.wrapper_model.regressor_
        
        importance = inner_model.feature_importances_
        feature_names = inner_model.feature_names_in_
        
        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)
        
        top_features = imp_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(
            x="Importance", y="Feature", data=top_features, 
            palette="viridis", hue="Feature", legend=False
        )
        plt.title(f"XGBoost (Log-Target) - Top {top_n} Feature Importance")
        plt.xlabel("Importance (Gain)")
        plt.tight_layout()
        
        if not os.path.exists("model/output"): os.makedirs("model/output")
        plt.savefig("model/output/feature_importance.png")
        print(green("Feature importance plot saved to 'model/output/feature_importance.png'"))

    def save_model(self, path="model/output/xgb_pisa_pipeline.pkl"):
        """
        Saves the entire pipeline (Wrapper + XGBoost) using Joblib.
        """
        if not os.path.exists("model/output"): os.makedirs("model/output")
        joblib.dump(self.wrapper_model, path)
        print(green(f"Model pipeline saved successfully to {path}"))

if __name__ == "__main__":
    DATA_DIR = "data" 
    # Auto-detect latest files logic
    import glob
    try:
        train_files = glob.glob(os.path.join(DATA_DIR, "X_train_clean_no_leakage_*.csv"))
        test_files = glob.glob(os.path.join(DATA_DIR, "X_test_clean_no_leakage_*.csv"))
        
        if train_files and test_files:
            X_TRAIN_PATH = max(train_files, key=os.path.getctime)
            X_TEST_PATH = max(test_files, key=os.path.getctime)
        else:
            # Fallback to defaults if clean files not found
            X_TRAIN_PATH = f"{DATA_DIR}/X_train_preprocessed_20251129_234423.csv" 
            X_TEST_PATH = f"{DATA_DIR}/X_test_preprocessed_20251129_234429.csv"
            
        Y_TRAIN_PATH = f"{DATA_DIR}/y_train.csv"

        xgb_pipeline = PisaXGBoost()

        # 1. LOAD
        X_train, y_train, _ = xgb_pipeline.load_data(X_TRAIN_PATH, Y_TRAIN_PATH)
        X_test, _, test_ids = xgb_pipeline.load_data(X_TEST_PATH, y_path=None)
        
        # 2. ALIGN (Crucial for test set)
        X_test = xgb_pipeline.align_features(X_train, X_test)
        
        # 3. TRAIN
        X_val, y_val = xgb_pipeline.train(X_train, y_train, val_size=0.1)
        
        # 4. SAVE & METRICS
        xgb_pipeline.save_model()
        xgb_pipeline.print_validation_metrics(X_val, y_val)

        # 5. PREDICT
        test_preds = xgb_pipeline.predict(X_test)
        
        # 6. SAVE SUBMISSION
        if test_ids is not None:
            submission_df = pd.DataFrame({"CNTSTUID": test_ids, "MathScore": test_preds})
        else:
            print(red("Warning: No IDs found in test set."))
            submission_df = pd.DataFrame({"MathScore": test_preds})
            
        submission_path = "model/output/submission_finetune.csv"
        if not os.path.exists("model/output"): os.makedirs("model/output")
        submission_df.to_csv(submission_path, index=False)
        print(green(f"Predictions saved to {submission_path}"))

        # 7. INTERPRET
        xgb_pipeline.plot_feature_importance(top_n=20)
        
    except Exception as e:
        print(red(f"Error during execution: {e}"))
        import traceback
        traceback.print_exc()