from helper_functions.print import *
from preprocessing.load import df_numerical

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
    numericals_null = df.null_count()
    cols_with_nulls = [
        col for col in numericals_null.columns 
        if numericals_null[col][0]/nb_raw > 0
    ]
    print(bold(f"remained null columns: \n {cols_with_nulls}"))
    
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
    to_drop = ["PA159", "IC183", "ST163", "FL164"]
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
    
    


if __name__ == "__main__":
    # Set up 
    nb_raw = df_numerical.shape[0]
    nb_col = df_numerical.shape[1]
    df_numerical = df_numerical.with_columns(
        pl.col(df_numerical.columns).cast(pl.Float64, strict=False)
    )
    
    # Clean too empty columns
    df_numerical = drop_sparse_columns(df_numerical)
    
    # Replace nll data by -1 on the score/timing
    df_numerical = handle_scores_and_timing(df_numerical)
    display_null(df_numerical)
    
    # process others
    df_numerical = clean_other(df_numerical)
    print_values_scope(df_numerical)
    print(df_numerical.shape)

   
    