import pandas as pd

chunksize = 50_000  # how many rows to load at a time
benign_rows = []
attack_rows = []
max_per_class = 50_000  # max number of rows you want per class (adjust as needed)

for chunk in pd.read_csv("merged_binary_dataset.csv", chunksize=chunksize):
    benign_chunk = chunk[chunk["Label"] == "Benign"]
    attack_chunk = chunk[chunk["Label"] != "Benign"]

    # Sample from current chunk (you can skip sampling if counts are already low)
    if len(benign_rows) < max_per_class:
        needed = max_per_class - len(benign_rows)
        benign_rows.append(benign_chunk.sample(n=min(needed, len(benign_chunk)), random_state=42))

    if len(attack_rows) < max_per_class:
        needed = max_per_class - len(attack_rows)
        attack_rows.append(attack_chunk.sample(n=min(needed, len(attack_chunk)), random_state=42))

    if len(benign_rows) >= max_per_class and len(attack_rows) >= max_per_class:
        break  # we have enough of both, stop early

# Combine and shuffle
df_balanced = pd.concat(benign_rows + attack_rows, ignore_index=True)
df_balanced = df_balanced.sample(frac=1, random_state=42)

df_balanced.to_csv("train_balanced_chunked.csv", index=False)
