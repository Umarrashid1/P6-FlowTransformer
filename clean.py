import pandas as pd


import csv

def clean_column_names(input_file, output_file):
    with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Read and clean header
        header = next(reader)
        cleaned_header = [col.replace('/', '_per_').replace(' ', '_') for col in header]
        writer.writerow(cleaned_header)

        # Copy the rest of the file
        for row in reader:
            writer.writerow(row)

    print(f"Cleaned file saved to {output_file}")


# Example usage:
input_file = 'merged_binary_dataset.csv'  # Replace with your input file path
output_file = 'merged_binary_dataset.csv'  # Replace with your desired output file path

clean_column_names(input_file, output_file)
