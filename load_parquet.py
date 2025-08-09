import pandas as pd

# Load parquet file into a DataFrame
df = pd.read_parquet("your_file.parquet")

print(df.head())
print(df.columns)
