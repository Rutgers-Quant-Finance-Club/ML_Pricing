import pandas as pd
import numpy as np

class DataPipeline:

    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

    def load(self):
        self.df = pd.read_csv(self.filepath)
        print(f"Loaded {self.df.shape[0]} rows and {self.df.shape[1]} columns")
        self.df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce', format = '%Y%m%d')
        # Company identified by PERMNO Number
        # Date YYYYMMDD
        self.df['year_month'] = self.df['DATE'].dt.to_period('M')
        # Keeping track of year and month
        return self

    def impute_missing(self):
        num_cols = self.df.select_dtypes(include='number').columns
        self.df[num_cols] = self.df[num_cols].fillna(
            self.df.groupby('year_month')[num_cols].transform('median')
        )
        # imputation of numeric columns using median of the respective month/year
        # following the direction of original paper
        # no non-numeric columns upon further investigation
        cleaned = self.df.dropna()
        
        print(f"Cleaned Data: {cleaned.df.shape[0]} rows and {cleaned.df.shape[1]} columns")
        self.df=cleaned
        
        return self
