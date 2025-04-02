import pandas as pd


def clean_column_names(input_file, output_file):
    # Read only the first row to get the column names (header)
    df = pd.read_csv(input_file, nrows=0)

    # Clean the column names by replacing '/' with '_per_' and ' ' with '_'
    cleaned_columns = [col.replace('/', '_per_').replace(' ', '_') for col in df.columns]

    # Open the input file and the output file
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        # Write the cleaned header to the output file
        header = infile.readline().strip().split(',')
        cleaned_header = [col.replace('/', '_per_').replace(' ', '_') for col in header]
        outfile.write(','.join(cleaned_header) + '\n')

        # Now, copy the remaining lines (data) from the input file to the output file
        for line in infile:
            outfile.write(line)

    print(f"Cleaned file saved to {output_file}")


# Example usage:
input_file = 'train_subset.csv'  # Replace with your input file path
output_file = 'train_subset_cleaned.csv'  # Replace with your desired output file path

clean_column_names(input_file, output_file)
