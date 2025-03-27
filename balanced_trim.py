import pandas as pd

chunksize = 1000
benign_rows = []
attack_rows = []
benign_count = 0
attack_count = 0
max_per_class = 10

for chunk in pd.read_csv("merged_binary_dataset.csv", chunksize=chunksize):
    if benign_count >= max_per_class and attack_count >= max_per_class:
        break

    benign_chunk = chunk[chunk["Label"] == "Benign"]
    attack_chunk = chunk[chunk["Label"] != "Benign"]

    if benign_count < max_per_class and not benign_chunk.empty:
        needed = max_per_class - benign_count
        sampled = benign_chunk.sample(n=min(needed, len(benign_chunk)), random_state=42)
        benign_rows.append(sampled)
        benign_count += len(sampled)

    if attack_count < max_per_class and not attack_chunk.empty:
        needed = max_per_class - attack_count
        sampled = attack_chunk.sample(n=min(needed, len(attack_chunk)), random_state=42)
        attack_rows.append(sampled)
        attack_count += len(sampled)

    print(f"Benign: {benign_count}, Attack: {attack_count}")

df_balanced = pd.concat(benign_rows + attack_rows, ignore_index=True)
df_balanced = df_balanced.sample(frac=1, random_state=42)
df_balanced.to_csv("train_balanced_chunked.csv", index=False)
