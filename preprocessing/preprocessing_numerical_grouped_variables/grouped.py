from helper_functions.print import *
from load import df_grouped_train
from numerical import print_values, print_values_scope, display_null

import polars as pl 
import polars.selectors as cs


def dataset_grouped_cleaner(df: pl.DataFrame):
    
    print(orange("-"*5 + " cleaning grouped dataset " + "-"*5))
    
    df_grouped_cleaned = df.with_columns(
            pl.col(df.columns).cast(pl.Float64, strict=False)
        ) # (1172086, 6)

    # Drop CNT and STRATUM, 100% of missing values 
    df_grouped_cleaned = df_grouped_cleaned.drop(["CNT", "STRATUM"])
    return df_grouped_cleaned

df_grouped_cleaned_train = dataset_grouped_cleaner(df_grouped_train)


if __name__ == "__main__":
    # Set up 
    nb_raw = df_grouped_train.shape[0]
    nb_col = df_grouped_train.shape[1]
    
    print(df_grouped_train.shape)
    
    df_grouped_cleaned_train = dataset_grouped_cleaner(df_grouped_train)
    
    print_values(df_grouped_cleaned_train)
    print_values_scope(df_grouped_cleaned_train)
    display_null(df_grouped_cleaned_train)