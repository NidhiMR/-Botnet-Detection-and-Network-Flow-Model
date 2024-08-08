import pandas as pd

# Load the CSV files
file1_path = 'flowdata4.binetflow.csv'
file2_path = 'preprocessed_flowdata4.csv'

file1 = pd.read_csv(file1_path)
file2 = pd.read_csv(file2_path)

# Create a dictionary to map integer labels to human-readable labels
label_mapping = {}

# Iterate through each row in the preprocessed file
for index, row in file2.iterrows():
    # Extract the integer label
    integer_label = row['label']
    
    # Find the corresponding row(s) in the unprocessed file
    match = file1[
        (file1['dur'] == row['dur']) &
        (file1['proto'] == row['proto']) &
        (file1['dir'] == row['dir']) &
        (file1['state'] == row['state']) &
        (file1['stos'] == row['stos']) &
        (file1['dtos'] == row['dtos']) &
        (file1['tot_pkts'] == row['tot_pkts']) &
        (file1['tot_bytes'] == row['tot_bytes']) &
        (file1['src_bytes'] == row['src_bytes'])
    ]
    
    if not match.empty:
        # Extract the human-readable label
        human_label = match.iloc[0]['label']
        
        # Map the integer label to the human-readable label
        label_mapping[integer_label] = human_label

# Print the mappings for 0 to 51
for i in range(52):
    label = label_mapping.get(i, 'N/A')
    print(f"{i}: {label}")
