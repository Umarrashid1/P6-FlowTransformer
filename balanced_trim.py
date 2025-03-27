import pandas as pd

df = pd.read_csv("merged_binary_dataset.csv")

# Define how much of each class to keep (e.g., 10% of each)
sample_frac = 0.1

# Group by Label and take a fraction from each
balanced_subset = df.groupby('Label', group_keys=False).apply(lambda x: x.sample(frac=sample_frac, random_state=42))

# Save the new subset
balanced_subset.to_csv("train_subset_balanced.csv", index=False)
