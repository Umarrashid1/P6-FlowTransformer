import pandas as pd

chunksize = 10000
target_rows = 100000
collected = []

for chunk in pd.read_csv("merged_binary_dataset.csv", chunksize=chunksize):
    collected.append(chunk)
    total = sum(len(c) for c in collected)
    if total >= target_rows:
        break

df = pd.concat(collected).sample(n=target_rows, random_state=42)
df.columns = [col.replace('/', '_per_').replace(' ', '_') for col in df.columns]
df.to_csv("train_subset.csv", index=False)
