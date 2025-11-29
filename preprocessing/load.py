from helper_functions.print import *

import polars as pl 
import polars.selectors as cs

def load_csv(path:str, display_columns_name:bool=False):
    print(orange("-"*10 + f" load : {path} " + "-"*10))
    df = pl.read_csv(path, ignore_errors=True)
    if display_columns_name:
        for i, (col_name, dtype) in enumerate(df.schema.items()):
            print(green("\n--- Column Types ---"))
            print(cyan(f"{col_name}: {dtype}"))
        
    return df

def select_numerical(df:pl.DataFrame):
    numerical = """AGE
    COBN_S
    WB165
    FL169
    WB166
    FL164
    ST355
    IC177
    IC183
    ST345
    PA182
    PA004
    ST059
    WB155
    ST163
    ST164
    WB177
    PA041
    ST175
    PA159
    ST165
    ST211
    WB176
    ST098
    PA002
    reading_q1_average_score
    reading_q2_average_score
    reading_q3_average_score
    reading_q4_average_score
    reading_q5_average_score
    reading_q6_average_score
    reading_q7_average_score
    reading_q8_average_score
    reading_q9_average_score
    reading_q10_average_score
    reading_q11_average_score
    reading_q12_average_score
    reading_q13_average_score
    reading_q14_average_score
    reading_q15_average_score
    math_q1_average_score
    math_q2_average_score
    math_q3_average_score
    math_q4_average_score
    math_q5_average_score
    math_q6_average_score
    math_q7_average_score
    math_q8_average_score
    math_q9_average_score
    math_q10_average_score
    math_q11_average_score
    math_q12_average_score
    math_q13_average_score
    math_q14_average_score
    math_q15_average_score
    math_q16_average_score
    math_q17_average_score
    math_q18_average_score
    math_q19_average_score
    math_q20_average_score
    math_q21_average_score
    science_q1_average_score
    science_q2_average_score
    science_q3_average_score
    science_q4_average_score
    science_q5_average_score
    science_q6_average_score
    science_q7_average_score
    science_q8_average_score
    science_q9_average_score
    science_q10_average_score
    science_q11_average_score
    science_q12_average_score
    science_q13_average_score
    science_q14_average_score
    science_q15_average_score
    science_q16_average_score
    science_q17_average_score
    science_q18_average_score
    science_q19_average_score
    reading_q1_total_timing
    reading_q2_total_timing
    reading_q3_total_timing
    reading_q4_total_timing
    reading_q5_total_timing
    reading_q6_total_timing
    reading_q7_total_timing
    reading_q8_total_timing
    reading_q9_total_timing
    reading_q10_total_timing
    reading_q11_total_timing
    reading_q12_total_timing
    reading_q13_total_timing
    reading_q14_total_timing
    reading_q15_total_timing
    math_q1_total_timing
    math_q2_total_timing
    math_q3_total_timing
    math_q4_total_timing
    math_q5_total_timing
    math_q6_total_timing
    math_q7_total_timing
    math_q8_total_timing
    math_q9_total_timing
    math_q10_total_timing
    math_q11_total_timing
    math_q12_total_timing
    math_q13_total_timing
    math_q14_total_timing
    math_q15_total_timing
    math_q16_total_timing
    math_q17_total_timing
    math_q18_total_timing
    math_q19_total_timing
    math_q20_total_timing
    math_q21_total_timing
    science_q1_total_timing
    science_q2_total_timing
    science_q3_total_timing
    science_q4_total_timing
    science_q5_total_timing
    science_q6_total_timing
    science_q7_total_timing
    science_q8_total_timing
    science_q9_total_timing
    science_q10_total_timing
    science_q11_total_timing
    science_q12_total_timing
    science_q13_total_timing
    science_q14_total_timing
    science_q15_total_timing
    science_q16_total_timing
    science_q17_total_timing
    science_q18_total_timing
    science_q19_total_timing
    """
    numerical = numerical.split()
    return df.select(numerical)

def select_grouped(df:pl.DataFrame):
    grouped = """
            Year
            CNT
            CNTRYID
            CNTSCHID
            CNTSTUID
            STRATUM
            """.split()
    return df.select(grouped)


df_train = load_csv("X_train.csv") # shape : (1172086, 307)
# y_test = load_csv("y_train.csv")
df_numerical_train = select_numerical(df_train) # shape : (1172086, 135) all columns include null values
df_grouped_train = select_grouped(df_train) # shape : (1172086, 6) all columns include null values

X_test = load_csv("X_test.csv")
df_numerical_test = select_numerical(X_test)
df_grouped_test = select_grouped(X_test)

if __name__ == "__main__":
    print(X_test.shape, df_train.shape)
    
    

 
    
        
    