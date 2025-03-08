import pandas as pd

# Define paths: update these paths as needed
input_csv = r"C:\Users\Pouya\Documents\university\Bachelor - AUT\Final Project\Data Set\Final Model Based Dataset\final_model_based_dataset.csv"
output_csv = r"C:\Users\Pouya\Documents\university\Bachelor - AUT\Final Project\Data Set\Final Model Based Binary\final_model_based_dataset_binary.csv"

# Load the dataset
df = pd.read_csv(input_csv)
print("Original dataset shape:", df.shape)

# Assuming the label column is the last column (or change 'labels' to your actual label column name)
label_col = df.columns[-1]

# Remove rows where the label is 0
df_binary = df[df[label_col] != 0]
print("Dataset shape after removing label 0:", df_binary.shape)

# Save the new filtered CSV
df_binary.to_csv(output_csv, index=False)
print("Filtered dataset saved to:", output_csv)
