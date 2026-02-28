import pandas as pd
import numpy as np

class DataPipeline:

    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

    def load(self):
        self.df = pd.read_csv(self.filepath)
        print(f"Loaded {self.df.shape[0]} rows and {self.df.shape[1]} columns")
        return self

    def impute_missing(self):
        return self