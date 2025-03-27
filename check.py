import pandas as pd

dataset_path = r"C:\Users\ronie\OneDrive - Aalborg Universitet\Desktop\P6-\P6-FlowTransformer\train_balanced_chunked.csv"
df = pd.read_csv(dataset_path)

# Print basic column stats
print("Column Name: 'Average Packet Size'")
print("Total Rows:", len(df))
print("Non-NaN Values:", df['Average Packet Size'].count())
print("First few values:\n", df['Average Packet Size'].head())

# Check if all values are NaN
if df['Average Packet Size'].dropna().empty:
    print("⚠️ The column is empty after dropping NaN values!")