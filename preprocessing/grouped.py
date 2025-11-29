from helper_functions.print import *
from preprocessing.load import df_grouped
from preprocessing.numerical import print_values, print_values_scope, display_null

import polars as pl 
import polars.selectors as cs

df_grouped = df_grouped.with_columns(
        pl.col(df_grouped.columns).cast(pl.Float64, strict=False)
    ) # (1172086, 6)

# Drop CNT and STRATUM, 100% of missing values 
    df_grouped = df_grouped.drop(["CNT", "STRATUM"])



if __name__ == "__main__":
    # Set up 
    nb_raw = df_grouped.shape[0]
    nb_col = df_grouped.shape[1]
    
    print(df_grouped.shape)
    
    df_grouped = df_grouped.with_columns(
        pl.col(df_grouped.columns).cast(pl.Float64, strict=False)
    ) # (1172086, 6)
    
    # Drop CNT and STRATUM, 100% of missing values 
    df_grouped = df_grouped.drop(["CNT", "STRATUM"])
    
    print_values(df_grouped)
    print_values_scope(df_grouped)
    display_null(df_grouped)