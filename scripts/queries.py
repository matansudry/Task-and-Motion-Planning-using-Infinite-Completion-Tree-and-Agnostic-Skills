import sys
sys.path.append(".")

import gzip
import pandas as pd


if __name__ == "__main__":
    path = "no_git/data_summary.gzip"
    df=pd.read_parquet(path)
    
    #count values
    print(df["previous_primitive"].value_counts())
    
    #query
    filtered_df = df.query('previous_primitive == "Pull"')
    