import pandas as pd

# Path to your original dataset
INPUT_CSV = 'merged_binary_dataset_cleaned.csv'

# Output path for the balanced dataset
OUTPUT_CSV = 'diad_balanced.csv'

# Read the dataset
print(f"Reading data from {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)

# Check that the required column exists
if 'Label' not in df.columns:
    raise ValueError("Expected 'Label' column not found in dataset.")

# Split into Benign and Attack
benign_df = df[df['Label'] == 'Benign']
attack_df = df[df['Label'] == 'Attack']

print(f"Benign samples: {len(benign_df)}, Attack samples: {len(attack_df)}")

# Match the number of Attack samples to the number of Benign ones
attack_sampled = attack_df.sample(n=len(benign_df), random_state=42)

# Combine and shuffle
balanced_df = pd.concat([benign_df, attack_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
print(f"Saving balanced dataset to {OUTPUT_CSV}...")
balanced_df.to_csv(OUTPUT_CSV, index=False)

print("✅ Done! New balanced dataset contains:")
print(balanced_df['Label'].value_counts())
