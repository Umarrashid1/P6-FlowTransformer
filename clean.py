import pandas as pd


# Define the function to clean column names
def clean_column_names(input_file, output_file):
    # Read the dataset from CSV
    df = pd.read_csv(input_file)

    # Clean column names by replacing '/' with '_per_' and ' ' with '_'
    df.columns = [col.replace('/', '_per_').replace(' ', '_') for col in df.columns]

    # Save the cleaned dataframe to a new CSV
    df.to_csv(output_file, index=False)
    print(f"Cleaned file saved to {output_file}")


# Example usage: clean column names in 'train_subset.csv' and save to 'train_subset_cleaned.csv'
input_file = 'merged_binary_dataset.csv'  # replace with your input file path
output_file = 'merged_binary_dataset.csv'  # replace with your desired output file path

clean_column_names(input_file, output_file)
