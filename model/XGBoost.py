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
        
        # Hyperparameters (Aggressive tuning for maximum R2)
        self.params = {
            'n_estimators': 8000,         # Increased to allow slow learning
            'learning_rate': 0.01,       # Slower & more precise
            'max_depth': 10,               # Deeper trees to capture complex interactions
            'min_child_weight': 5,        # Lowered to allow learning from smaller groups
            'subsample': 0.85,            
            'colsample_bytree': 0.8,     
            'gamma': 0.2,                 # Reduced pruning (let trees grow)
            'reg_alpha': 0.5,             # Reduced L1 (allow more features)
            'reg_lambda': 1.0,            # Reduced L2 (allow larger weights)
            'n_jobs': -1,                 
            'random_state': self.random_state,
            'objective': 'reg:squarederror', 
            'tree_method': 'hist',
            'early_stopping_rounds': 150   
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
            
        # 2. Drop Non-Feature columns
        cols_to_drop = ["student_id", "Unnamed: 0", "CNTSTUID", "CNTRYID", "CNTSCHID", "MathScore"]
        X = X.drop(columns=cols_to_drop, errors='ignore')
        
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
            valid_mask = y > 0.0
            if not valid_mask.all():
                initial_count = len(y)
                X = X.loc[valid_mask]
                y = y[valid_mask]
                print(red(f"Dropped {initial_count - len(y)} rows where Target <= 0.0"))

            print(bold(f"Final Target Stats -> Min: {y.min():.2f}, Max: {y.max():.2f}, Mean: {y.mean():.2f}"))

        return X, y, ids

    def add_interactions(self, df):
        """
        Re-introduced: Adds Squared Features to capture non-linear returns.
        """
        print(orange("Generating Polynomial Features (x^2)..."))
        # Focus on continuous variables where "more" might have diminishing returns
        target_keywords = ['AGE', 'timing', 'efficiency', 'ESCS', 'wealth', 'home'] 
        cols_to_square = [c for c in df.columns if any(k in c.lower() for k in target_keywords)]
        
        count = 0
        for col in cols_to_square:
            if pd.api.types.is_numeric_dtype(df[col]):
                new_col_name = f"{col}_sq"
                # Square the column
                sq_vals = df[col] ** 2
                
                # Check for Infinity/Overflow
                if np.isinf(sq_vals).any() or (sq_vals > 1e15).any():
                    continue # Skip if it creates dangerous values
                    
                df[new_col_name] = sq_vals
                count += 1
        
        print(f"Added squared features for {count} columns.")
        return df

    def align_features(self, X_train, X_test):
        print(orange("\n--- Aligning Test Features with Training Features ---"))
        train_cols = X_train.columns.tolist()
        test_cols = X_test.columns.tolist()
        
        missing = [c for c in train_cols if c not in test_cols]
        if missing:
            print(red(f"Warning: X_test missing {len(missing)} columns. Filling with -1."))
            for c in missing:
                X_test[c] = -1
            
        X_test = X_test[train_cols]
        return X_test

    def train(self, X_train, y_train, val_size=0.1):
        print(orange(f"\n--- Starting XGBoost Training with Features + TransformedTarget ---"))
        
        # 1. Feature Engineering (Polynomials)
        X_train = self.add_interactions(X_train)
        self.train_features_ = X_train.columns.tolist()
        
        # 2. Split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, 
            test_size=val_size, 
            random_state=self.random_state
        )

        # 3. Initialize Inner Regressor
        xgb_inner = xgb.XGBRegressor(**self.params)
        
        # 4. Initialize Wrapper (Log-Transform Target)
        self.wrapper_model = TransformedTargetRegressor(
            regressor=xgb_inner,
            func=np.log1p,
            inverse_func=np.expm1
        )
        
        # 5. Fit
        print(orange("Fitting model (Log-Transform applied internally)..."))
        
        # Manual transform for eval_set
        y_val_transformed = np.log1p(y_val)
        
        self.wrapper_model.fit(
            X_tr, y_tr,
            eval_set=[(X_tr, np.log1p(y_tr)), (X_val, y_val_transformed)],
            verbose=200
        )
        
        print(green("Training Complete."))
        return X_val, y_val

    def print_validation_metrics(self, X_val, y_val):
        print(orange("\n--- Final Validation Metrics ---"))
        
        preds = self.wrapper_model.predict(X_val)
        
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)
        
        print(bold(f"Validation RMSE: {rmse:,.4f}"))
        print(f"Validation MAE:  {mae:,.4f}")
        print(f"Validation R2:   {r2:.4f}")

    def predict(self, X_test):
        print(orange("\n--- Generating Predictions ---"))
        
        # Apply same feature engineering
        X_test = self.add_interactions(X_test)
        # Reindex to match training columns exactly
        X_test = X_test.reindex(columns=self.train_features_, fill_value=-1)
        
        predictions = self.wrapper_model.predict(X_test)
        
        # Final safety clip
        predictions = np.clip(predictions, 0, 1000)
        
        p_min, p_max, p_mean = predictions.min(), predictions.max(), predictions.mean()
        print(bold(f"Prediction Stats -> Min: {p_min:.2f}, Max: {p_max:.2f}, Mean: {p_mean:.2f}"))
        
        return predictions

    def plot_feature_importance(self, top_n=20):
        print(orange("\n--- Plotting Feature Importance ---"))
        if self.wrapper_model is None: return

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
        plt.title(f"XGBoost (Optimized) - Top {top_n} Feature Importance")
        plt.xlabel("Importance (Gain)")
        plt.tight_layout()
        
        if not os.path.exists("model/output"): os.makedirs("model/output")
        plt.savefig("model/output/feature_importance.png")
        print(green("Feature importance plot saved to 'model/output/feature_importance.png'"))

    def save_model(self, path="model/output/xgb_pisa_pipeline.pkl"):
        if not os.path.exists("model/output"): os.makedirs("model/output")
        joblib.dump(self.wrapper_model, path)
        print(green(f"Model pipeline saved successfully to {path}"))

if __name__ == "__main__":
    DATA_DIR = "data" 
    X_TRAIN_PATH = f"{DATA_DIR}/X_train_preprocessed_20251130_114328.csv" 
    Y_TRAIN_PATH = f"{DATA_DIR}/y_train.csv"
    X_TEST_PATH = f"{DATA_DIR}/X_test_preprocessed_20251130_114332.csv"   

    xgb_pipeline = PisaXGBoost()

    try:
        X_train, y_train, _ = xgb_pipeline.load_data(X_TRAIN_PATH, Y_TRAIN_PATH)
        X_test, _, test_ids = xgb_pipeline.load_data(X_TEST_PATH, y_path=None)
        
        X_val, y_val = xgb_pipeline.train(X_train, y_train, val_size=0.1)
        
        xgb_pipeline.save_model()
        xgb_pipeline.print_validation_metrics(X_val, y_val)

        test_preds = xgb_pipeline.predict(X_test)
        
        if test_ids is not None:
            submission_df = pd.DataFrame({"ID": test_ids, "MathScore": test_preds})
        else:
            print(red("Warning: No IDs found in test set."))
            submission_df = pd.DataFrame({"MathScore": test_preds})
            
        submission_path = "model/output/submission_finetune.csv"
        if not os.path.exists("model/output"): os.makedirs("model/output")
        submission_df.to_csv(submission_path, index=False)
        print(green(f"Predictions saved to {submission_path}"))

        xgb_pipeline.plot_feature_importance(top_n=20)
        
    except Exception as e:
        print(red(f"Error during execution: {e}"))
        import traceback
        traceback.print_exc()