import pandas as pd

# Load full dataset into memory
df = pd.read_csv("merged_binary_dataset.csv")

# Normalize label format
df["Label"] = df["Label"].astype(str).str.strip().str.lower()

# Split into classes
benign_df = df[df["Label"] == "benign"]
attack_df = df[df["Label"] != "benign"]

print(f"✅ Found {len(benign_df)} benign rows and {len(attack_df)} attack rows.")

# Get equal number of rows from each class (based on the smaller one)
n_samples = min(len(benign_df), len(attack_df))
print(f"✅ Sampling {n_samples} rows per class to balance.")

# Sample randomly (reproducible with seed)
benign_sample = benign_df.sample(n=n_samples, random_state=42)
attack_sample = attack_df.sample(n=n_samples, random_state=42)

# Combine and shuffle
df_balanced = pd.concat([benign_sample, attack_sample], ignore_index=True)
df_balanced = df_balanced.sample(frac=1, random_state=42)

# Save to file
df_balanced.to_csv("train_balanced_fullmem.csv", index=False)
print("✅ Balanced dataset saved as 'train_balanced_fullmem.csv'")
