import numpy as np
import pandas as pd

# Load the saved .csv files

data = pd.read_csv('preprocessed_stock_data.csv')

# get the name of all columns
print(data.columns)

