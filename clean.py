import csv
import os

def clean_column_names(input_file, output_file):
    if os.stat(input_file).st_size == 0:
        print(f"Input file '{input_file}' is empty. No output written.")
        return

    with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        try:
            header = next(reader)
        except StopIteration:
            print(f"Input file '{input_file}' has no header row. No output written.")
            return

        cleaned_header = [col.replace('/', '_per_').replace(' ', '_') for col in header]
        writer.writerow(cleaned_header)

        for row in reader:
            writer.writerow(row)

    print(f"Cleaned file saved to {output_file}")



# Example usage:
input_file = 'merged_binary_dataset.csv'  # Replace with your input file path
output_file = 'merged_binary_dataset.csv'  # Replace with your desired output file path

clean_column_names(input_file, output_file)
