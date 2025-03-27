import os
import pandas as pd

data_dir = '\\\\wsl.localhost\\Ubuntu\\home\\ubuntu\\DatasetFlow'
dfs = []

for root, _, files in os.walk(data_dir):  # walks through subdirs too
    for filename in files:
        if filename.endswith('.csv'):
            file_path = os.path.join(root, filename)
            df = pd.read_csv(file_path)

            # Decide if this is benign or attack based on the folder or file name
            if 'benign' in file_path.lower():
                df['Label'] = 'Benign'
            else:
                df['Label'] = 'Attack'

            dfs.append(df)

full_dataset = pd.concat(dfs, ignore_index=True)
full_dataset.to_csv("merged_binary_dataset.csv", index=False)