import pandas as pd
import numpy as np
import os
import glob

# Helper for colored printing
try:
    from preprocessing.helper_functions.print import orange, green, red, bold
except ImportError:
    def orange(x): return x
    def green(x): return x
    def red(x): return x
    def bold(x): return x

def remove_leakage_and_sparse(x_train_path, x_test_path):
    """
    Loads X_train and X_test.
    1. Removes 'Math' sub-scores and timings (Leakage).
    2. Removes columns with > 95% missing values.
    3. Aligns columns between Train and Test.
    """
    print(orange(f"Loading datasets..."))
    print(f"Train: {x_train_path}")
    print(f"Test:  {x_test_path}")
    
    df_train = pd.read_csv(x_train_path)
    df_test = pd.read_csv(x_test_path)
    
    print(bold(f"\nInitial Shapes -> Train: {df_train.shape}, Test: {df_test.shape}"))

    # --- 1. REMOVE MATH LEAKAGE ---
    # We look for columns containing "math" AND ("average_score" or "total_timing")
    leakage_cols = [
        c for c in df_train.columns 
        if 'math' in c.lower() and ('average_score' in c.lower() or 'total_timing' in c.lower())
    ]
    
    if leakage_cols:
        print(orange(f"\nRemoving {len(leakage_cols)} Math Leakage columns..."))
        # print(leakage_cols) # Uncomment to see names
        df_train = df_train.drop(columns=leakage_cols)
        # Only drop from test if they exist there
        existing_test_leakage = [c for c in leakage_cols if c in df_test.columns]
        df_test = df_test.drop(columns=existing_test_leakage)
        
    # --- 2. REMOVE SPARSE COLUMNS (> 95% NaN) ---
    threshold = 0.95
    # Calculate null percentage on TRAIN only
    null_counts = df_train.isnull().mean()
    sparse_cols = null_counts[null_counts > threshold].index.tolist()
    
    if sparse_cols:
        print(orange(f"\nRemoving {len(sparse_cols)} Sparse columns (> {threshold:.0%})..."))
        df_train = df_train.drop(columns=sparse_cols)
        # Drop same columns from test
        existing_test_sparse = [c for c in sparse_cols if c in df_test.columns]
        df_test = df_test.drop(columns=existing_test_sparse)

    # --- 3. ALIGN COLUMNS ---
    print(orange("\nAligning Train and Test columns..."))
    
    # Get columns present in Train
    train_cols = df_train.columns.tolist()
    
    # 1. Drop extra columns in Test (not in Train)
    test_extras = [c for c in df_test.columns if c not in train_cols]
    if test_extras:
        print(f"Dropping {len(test_extras)} extra columns from Test.")
        df_test = df_test.drop(columns=test_extras)
        
    # 2. Add missing columns in Test (present in Train) - Fill with -1 or 0
    test_missing = [c for c in train_cols if c not in df_test.columns]
    if test_missing:
        print(f"Adding {len(test_missing)} missing columns to Test (filled with -1).")
        for c in test_missing:
            df_test[c] = -1
            
    # 3. Enforce Order
    df_test = df_test[train_cols]
    
    # --- 4. FINAL CHECK ---
    print(bold("\nFinal Verification:"))
    print(f"Train Columns: {df_train.shape[1]}")
    print(f"Test Columns:  {df_test.shape[1]}")
    
    if df_train.shape[1] == df_test.shape[1]:
        print(green("SUCCESS: Column counts match."))
    else:
        print(red("ERROR: Column counts do not match!"))
        
    print(f"Train Rows: {df_train.shape[0]}")
    print(f"Test Rows:  {df_test.shape[0]}")

    return df_train, df_test

if __name__ == "__main__":
    # CONFIGURATION
    DATA_DIR = "data"
    
    # AUTOMATICALLY FIND LATEST PREPROCESSED FILES
    try:
        train_files = glob.glob(os.path.join(DATA_DIR, "X_train_preprocessed_*.csv"))
        test_files = glob.glob(os.path.join(DATA_DIR, "X_test_preprocessed_*.csv"))
        
        if not train_files or not test_files:
            raise ValueError("No preprocessed files found.")

        # Sort by creation time to get latest
        X_TRAIN_PATH = max(train_files, key=os.path.getctime)
        X_TEST_PATH = max(test_files, key=os.path.getctime)
        
        # EXECUTE CLEANING (No y_train passed)
        df_train_clean, df_test_clean = remove_leakage_and_sparse(X_TRAIN_PATH, X_TEST_PATH)
        
        # SAVE
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        train_out = os.path.join(DATA_DIR, f"X_train_clean_no_leakage_{timestamp}.csv")
        test_out = os.path.join(DATA_DIR, f"X_test_clean_no_leakage_{timestamp}.csv")
        
        print(orange(f"\nSaving cleaned files..."))
        df_train_clean.to_csv(train_out, index=False)
        df_test_clean.to_csv(test_out, index=False)
        print(green(f"Saved: {train_out}"))
        print(green(f"Saved: {test_out}"))
        
    except ValueError as e:
        print(red(f"Error: {e} Check paths in 'data/'."))