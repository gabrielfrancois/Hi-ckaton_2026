from helper_functions.print import *
from load import df_numerical_train

import polars as pl 
import polars.selectors as cs

def drop_sparse_columns(df: pl.DataFrame, threshold: float = 0.8):
    """
    Drops columns that have a null rate higher than the threshold.
    """
    print(orange(f"\n--- Dropping Sparse Columns (Threshold: {threshold:.0%}) ---"))
    
    null_counts = df.null_count()
    total_rows = df.height
    numerical_cols_with_nulls = [
        col for col in null_counts.columns 
        if null_counts[col][0]/total_rows > threshold
    ]
    print(orange(f"    more than {threshold:.0%} of missing data : {len(numerical_cols_with_nulls)} columns, representing {len(numerical_cols_with_nulls)/df.shape[1]:.0%} of our dataset: \n"))
    
    to_drop = []

    # Iterate through the null_counts (which is a 1-row DataFrame)
    for col in df.columns:
        # Check null count for this specific column
        count = null_counts[col][0]
        if (count / total_rows) > threshold:
            to_drop.append(col)
    
    if to_drop:
        print(f"    Dropping {len(to_drop)} columns: {to_drop}")
        return df.drop(to_drop)
    else:
        print("    No columns exceeded the null threshold.")
        return df

def print_values(df=pl.DataFrame):
    """
    print the values of each columns
    """
    for col in df.columns:
        # value_counts() works on any data type and includes nulls
        dist = df[col].value_counts(sort=True)
        # Use pl.Config context manager to force printing ALL rows (tbl_rows=-1)
        # This ensures the output is not truncated, no matter how long the list is.
        if dist.shape[0] < 10 :
            with pl.Config(tbl_rows=-1):
                print(dist)
        else:
            print(dist)

def print_values_scope(df=pl.DataFrame):
    """
    print the values of each columns
    """
    print(orange("\n--- Column Value Scopes (Min/Max) ---"))
    for col in df.columns:
        try:
            min_val = df[col].min()
            max_val = df[col].max()
            med_val = df[col].median()
            mean_val = df[col].mean()
            # formatting: <30 ensures the column name takes up 30 chars
            if int(max_val) > 7200: 
                print(red(f"{col:<30} | Min: {str(min_val)} | Max: {str(max_val)} | median: {str(med_val)} | mean: {str(mean_val)}"))
            else:
                print(f"{col:<30} | Min: {str(min_val)} | Max: {str(max_val)} | median: {str(med_val)} | mean: {str(mean_val)}")
            
        except Exception as e:
            print(f"Col: {col:<30} | Error: {e}")

def display_null(df:pl.DataFrame):
    """
    Just print the columns that include null value.
    """
    nb_raw = df.shape[0]
    numericals_null = df.null_count()
    cols_with_nulls = [
        col for col in numericals_null.columns 
        if numericals_null[col][0]/nb_raw > 0
    ]
    if cols_with_nulls:
        print(bold(f"Remained null columns: \n {cols_with_nulls}"))
    else:
        print(bold("No column containing null value left"))
    
def handle_scores_and_timing(df: pl.DataFrame, cap_seconds:int=1800):
    """
    1. Selects all columns with 'score' in the name -> Fills nulls with -1
    2. Selects all columns with 'timing' in the name -> Fills nulls with -1
    3. Converts milliseconds to seconds.
    4. CAPS (Winsorizes) outliers at 'cap_seconds' (default 30 mins).
    """
    print(orange("\n--- Handling Missing Values (Imputation) ---"))

    print("    Filling missing 'score' columns with -1...")
    df = df.with_columns(
        pl.col("^.*score.*$").fill_null(-1)
    )

    print("    Filling missing 'timing' columns with -1...")
    df = df.with_columns(
        pl.col("^.*timing.*$").fill_null(-1)
    )
    
    # CAPS (Winsorizes) outliers at 'cap_seconds' (default 30 mins)
    timing_cols = cs.matches("^.*timing.*$")
    df = df.with_columns(
        pl.when(timing_cols != -1)
        .then(
            timing_cols
            .truediv(1000)        # Divide by 1000
            .clip(upper_bound=cap_seconds) # Cap extreme outliers 
        )
        .otherwise(-1)
        .name.keep()
    )
    
    return df   

def clean_other(df: pl.DataFrame, threshold: float = 0.05):
    """
    Scans all remaining numeric columns.
    - delete useless/redundant data
    - If null rate < threshold (e.g. 5%): Replace with MEDIAN.
    - If null rate >= threshold: Replace with -1.
    """
    
    print(orange(f"\n--- Handling Remaining Numeric Nulls (Threshold: {threshold:.0%}) ---"))
    
    # PA159 = ST175 ! IC183 and ST163 : correlation != reason
    # FL164 --> no sens
    # "WB177" (prefr WB176), "PA041" (correlation != reason), "ST164" (same), "ST165" (too correlated with ST059), "ST211" (too corelated with PA182), PA182 (same as ST059) decided after correlation matrix
    to_drop = ["PA159", "IC183", "ST163", "FL164", "WB177", "PA041", "ST164", "ST165", "ST211", "PA182", "ST175"]
    print(f"    delete useless data: {to_drop}")
    df = df.drop(to_drop)
    
    null_counts = df.null_count()
    total_rows = df.height
    
    cols_median = []
    cols_flag = []
    
    # Check only numeric columns that still have nulls
    numeric_cols = df.select(cs.numeric()).columns
    
    for col in numeric_cols:
        count = null_counts[col][0]
        if count == 0: continue # Skip fully clean columns
        
        rate = count / total_rows
        
        if rate < threshold:
            cols_median.append(col)
        else:
            cols_flag.append(col)
            
    # Apply Median Fill
    if cols_median:
        print(f"    Filling {len(cols_median)} columns with MEDIAN (low missing rate): {cols_median}")
        # We compute median for each column individually
        df = df.with_columns([
            pl.col(c).fill_null(pl.col(c).median()) for c in cols_median
        ])

    # Apply -1 Fill
    if cols_flag:
        print(f"    Filling {len(cols_flag)} columns with -1 (high missing rate): {cols_flag}")
        df = df.with_columns([
            pl.col(c).fill_null(-1) for c in cols_flag
        ])
        
    return df

def check_high_correlations(df: pl.DataFrame, threshold: float = 0.90):
    """
    Calculates the correlation matrix and prints pairs with correlation > threshold.
    """
    print(orange(f"\n--- Checking Correlations > {threshold} (Redundancy Check) ---"))
    
    # Select only numeric columns
    numeric_df = df.select(cs.numeric())
    
    # Calculate Correlation Matrix
    corr_df = numeric_df.corr()
    
    # Iterate to find high values
    cols = corr_df.columns
    printed_pairs = set()
    found_high_corr = False
    
    for i, col_a in enumerate(cols):
        for j, col_b in enumerate(cols):
            if i >= j: continue # Skip diagonal and duplicates
            
            # Polars correlation matrix can be accessed by column name and row index
            val = corr_df[col_a][j] 
            
            if abs(val) > threshold:
                found_high_corr = True
                print(f"High Correlation ({val:.4f}): {col_a} <--> {col_b}")
                
    if not found_high_corr:
        print("No columns found exceeding the redundancy threshold.")

# TODO actually don't use it but it can be enhanced to become relevant!
def create_interaction_features(df: pl.DataFrame):
    """
    Creates Efficiency (Score/Time) and Fast Guesser flags.
    Goal : reduce explainable variable to prevent overfitting
    """
    print(orange("\n--- Creating Interaction Features (Efficiency & Fast Guesser) ---"))
    
    new_cols_exprs = []
    cols_to_drop = []
    
    # Iterate over all columns to find 'average_score'
    for col in df.columns:
        if "average_score" in col:
            # Infer the matching timing column name
            timing_col = col.replace("average_score", "total_timing")
            guesser_name = col.replace("average_score", "is_fast_guesser")
            
            # Only proceed if the matching timing column actually exists
            if timing_col in df.columns:
                eff_name = col.replace("average_score", "efficiency")
                
                # Logic: Efficiency = Score / Time
                # If time is valid (>0), divide. Otherwise set efficiency to -1.
                eff_expr = (
                    pl.when(pl.col(timing_col) > 0)
                    .then(pl.col(col) / pl.col(timing_col))
                    .otherwise(-1) 
                    .alias(eff_name)
                )
                
                # Assumes timing is already normalized to seconds
                guesser_expr = (
                    pl.when((pl.col(timing_col) > 0) & (pl.col(timing_col) < 5))
                    .then(1.0) # Using float for consistency, or 1 for int
                    .otherwise(0.0)
                    .alias(guesser_name)
                )
                
                new_cols_exprs.append(guesser_expr)
                new_cols_exprs.append(eff_expr)
                
                # Mark originals for deletion
                cols_to_drop.append(col)
                cols_to_drop.append(timing_col)
                
    if new_cols_exprs:
        print(f"Creating {len(new_cols_exprs)} efficiency columns and dropping {len(cols_to_drop)} original columns...")
        
        # Add the new efficiency columns
        df = df.with_columns(new_cols_exprs)
        
        # Drop the old source columns
        df = df.drop(cols_to_drop)
    else:
        print("No matching Score/Time pairs found to replace.")
        
    return df

# only use it for XGBoost
def normalize_preserving_flags(df: pl.DataFrame, flag_value: float = -1.0):
    """
    Normalizes columns to [0, 1] range via MinMax Scaling, 
    BUT ignores and preserves the 'flag_value' (e.g., -1).
    Formula: X_norm = (X - X_min) / (X_max - X_min)
    """
    print(orange("\n--- Normalizing Data (Keeping -1 as -1) ---"))
    
    numeric_cols = df.select(cs.numeric()).columns
    expressions = []
    
    for col in numeric_cols:
        # Compute min and max EXCLUDING the flag
        valid_data = df.select(pl.col(col).filter(pl.col(col) != flag_value))
        
        if valid_data.height > 0:
            min_val = valid_data.item(0, 0) # Just grabbing the first item isn't min, need .min()
            min_val = valid_data.select(pl.min(col)).item()
            max_val = valid_data.select(pl.max(col)).item()
            
            # Avoid division by zero if max == min
            if max_val != min_val:
                # Apply normalization only where value != flag
                expr = (
                    pl.when(pl.col(col) != flag_value)
                    .then((pl.col(col) - min_val) / (max_val - min_val))
                    .otherwise(flag_value) # Keep the flag
                    .alias(col)
                )
                expressions.append(expr)
    
    if expressions:
        print(f"    Normalizing {len(expressions)} columns...")
        df = df.with_columns(expressions)
        
    return df

def dataset_numerical_cleaner(df: pl.DataFrame):
    
    print(orange("-"*5 + " cleaning numerical dataset " + "-"*5))

    df_numerical_cleaned = df_numerical_train.with_columns(
            pl.col(df_numerical_train.columns).cast(pl.Float64, strict=False)
        )

    # # Clean too empty columns
    df_numerical_cleaned = drop_sparse_columns(df_numerical_cleaned)

    # # Replace nll data by -1 on the score/timing
    df_numerical_cleaned = handle_scores_and_timing(df_numerical_cleaned)

    # # process others
    df_numerical_cleaned = clean_other(df_numerical_cleaned)

    # # normalize data
    # df_numerical_cleaned = normalize_preserving_flags(df_numerical_cleaned)
    
    return df_numerical_cleaned
    
    
df_numerical_cleaned_train = dataset_numerical_cleaner(df_numerical_train)


if __name__ == "__main__":
    # Set up 
    nb_raw = df_numerical_train.shape[0]
    nb_col = df_numerical_train.shape[1]
    
    df_numerical_cleaned_train = dataset_numerical_cleaner(df_numerical_train)
        
    check_high_correlations(df_numerical_cleaned_train)
    print(df_numerical_cleaned_train.shape)
    

   
    