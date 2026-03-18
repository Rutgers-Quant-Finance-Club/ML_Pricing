import pandas as pd
import numpy as np

class DataPipeline:

    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

    def load(self):
        # load csv into a pd dataframe; cast values to np.float32's for memory purposes; turn values in date to proper datetime objects; sort rows by permno, and inside permno groups, date
        self.df = pd.read_csv(self.filepath, dtype={col: np.float32 for col in pd.read_csv(self.filepath, nrows=0).columns if col != 'DATE'})
        self.df['DATE'] = pd.to_datetime(self.df['DATE'].astype(str).str.split('.').str[0], format='%Y%m%d')
        self.df = self.df.sort_values(['permno', 'DATE']).reset_index(drop=True)

        # create a return column using 'mom1m' as a proxy; 'mom1m' is the current stocks value - so, shifting each mom1m by -1 lets return equal the NEXT months mom1m
        self.df['ret'] = self.df.groupby('permno')['mom1m'].shift(-1)

        
        # create a new column such that it is the date with day truncated
        self.df['year_month'] = self.df['DATE'].dt.to_period('M')

        print(f"Memory: {self.df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
        print(f"Loaded {self.df.shape[0]} rows and {self.df.shape[1]} columns")

        return self

    def impute_missing(self):
        num_cols = self.df.select_dtypes(include='number').columns
        # for all numerical columnms, fill NA values with the median of that columns feature with respect to all other rows in the same year_month
        for col in num_cols:
            self.df[col] = self.df[col].fillna(
                self.df.groupby('year_month')[col].transform('median')
            )
        # imputation of numeric columns using median of the respective month/year
        # following the direction of original paper
        # no non-numeric columns upon further investigation
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        print(f"Cleaned Data: {self.df.shape[0]} rows and {self.df.shape[1]} columns")
        return self

    def normalize(self):
        exclude = ['DATE', 'year_month', 'permno', 'ret']  
        # for all numerical columns except those in exclude, normalize them to values between -1 and 1 inclusive to avoid heavy outliers
        num_cols = [c for c in self.df.select_dtypes(include='number').columns if c not in exclude]
        for col in num_cols:
            self.df[col] = self.df.groupby('year_month')[col].transform(
                lambda x: (x.rank(method='average') - 1) / (len(x) - 1) * 2 - 1
            )
        print("Features normalized to [-1, 1]")
        return self
        # cross-sectional rank normalization