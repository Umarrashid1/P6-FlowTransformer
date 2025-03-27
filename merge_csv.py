import os
import pandas as pd

data_dir = r'\\wsl.localhost\Ubuntu\home\ubuntu\DatasetFlow'
output_file = 'merged_binary_dataset.csv'
first = True  # To handle writing header only once

for root, _, files in os.walk(data_dir):
    for filename in files:
        print(filename)
        if filename.endswith('.csv'):
            file_path = os.path.join(root, filename)

            try:
                df = pd.read_csv(file_path)

                if 'benign' in file_path.lower():
                    df['Label'] = 'Benign'
                else:
                    df['Label'] = 'Attack'

                # Append to output CSV without keeping all in memory
                df.to_csv(output_file, mode='a', index=False, header=first)
                first = False  # Only write header once

            except Exception as e:
                print(f"Error reading {file_path}: {e}")
