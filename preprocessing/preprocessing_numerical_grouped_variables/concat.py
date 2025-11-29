from .grouped import df_grouped_cleaned_train, df_grouped_cleaned_test
from .numerical import df_numerical_cleaned_train, df_numerical_cleaned_test
from ..helper_functions.print import *
from .load import df_numerical_train, df_grouped_train, df_train, df_numerical_test, df_grouped_test, X_test, DATA_DIR

import polars as pl
import polars.selectors as cs


def rebuild_original(df_global : pl.DataFrame, df_grouped_cleaned: pl.DataFrame, df_numerical_cleaned: pl.DataFrame, df_grouped: pl.DataFrame, df_numerical: pl.DataFrame):
    """
    build a new X_train with the new preprocessed data from df_cleaned and the other columns 
    (excluding the numerical and grouped not kept) and X_train.
    Args:
        df_global (pl.DataFrame): the original dataset 
        df_grouped_cleaned:  the new preprocessed of grouped variables
        df_numerical_cleaned: the new preprocessed of numerical variables
        df_grouped: original df_grouped
        df_numerical: original df_numerical
    return:
        df_preprocessed : df_global with  the processed variable modifyed
    """
    
    print(orange("-"*5 + " Rebuilding Global Dataset " + "-"*5))

    # Identify ALL columns to remove from Global (The raw versions)
    cols_to_remove = set(df_grouped.columns) | set(df_numerical.columns)
    
    # Strip Global
    df_global_stripped = df_global.drop(list(cols_to_remove))
    
    print(f"    Original Global Shape: {df_global.shape}")
    print(f"    Global after stripping raw columns: {df_global_stripped.shape}")

    # Handle Duplicate Columns before Concat
    # Polars horizontal concat will fail if multiple dataframes have the same column name (e.g., 'student_id').
    
    # Check Numerical against Global
    common_num = set(df_numerical_cleaned.columns).intersection(set(df_global_stripped.columns))
    if common_num:
        print(red("Got some trouble, numerical and global shared the same columns ! Likely an error occured..." ))
        df_numerical_cleaned = df_numerical_cleaned.drop(list(common_num))
        
    # Check Grouped against Global AND Numerical
    existing_cols = set(df_global_stripped.columns) | set(df_numerical_cleaned.columns)
    common_grp = set(df_grouped_cleaned.columns).intersection(existing_cols)
    if common_grp:
        df_grouped_cleaned = df_grouped_cleaned.drop(list(common_grp))

    # Concatenate (Horizontal)
    df_preprocessed = pl.concat(
        [df_global_stripped, df_numerical_cleaned, df_grouped_cleaned], 
        how="horizontal"
    )
    
    print(f"Final Rebuilt Shape: {df_preprocessed.shape}")
    return df_preprocessed
    

if __name__ == "__main__":
    X_numerical_grouped_cleaned_train = rebuild_original(df_global=df_train,
                                                         df_grouped_cleaned= df_grouped_cleaned_train,
                                                         df_numerical_cleaned=df_numerical_cleaned_train,
                                                         df_grouped=df_grouped_train,
                                                         df_numerical=df_numerical_train
                                                         )
    output_filename = str(DATA_DIR / "X_numerical_grouped_cleaned_train.csv")
    print(orange("-"*10 + f" Saving to {output_filename} " + "-"*10))
    X_numerical_grouped_cleaned_train.write_csv(output_filename)
    print(green(f"Saved {output_filename} successfully!"))
    X_numerical_grouped_cleaned_test= rebuild_original(df_global=X_test,
                                                         df_grouped_cleaned= df_grouped_cleaned_test,
                                                         df_numerical_cleaned=df_numerical_cleaned_test,
                                                         df_grouped=df_grouped_test,
                                                         df_numerical=df_numerical_test
                                                         )
    output_filename = str(DATA_DIR / "X_numerical_grouped_cleaned_test.csv")
    print(orange("-"*10 + f" Saving to {output_filename} " + "-"*10))
    X_numerical_grouped_cleaned_test.write_csv(output_filename)
    print(green(f"Saved {output_filename} successfully!"))