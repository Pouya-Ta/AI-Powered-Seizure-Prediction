import os
import pandas as pd

print("Step 1: Load the sorted feature scores CSV")
############################################
# Step 1: Load the sorted feature scores CSV
############################################
feature_scores_path = r"C:\Users\Pouya\Documents\university\Bachelor - AUT\Final Project\Data Set\Selected_Features\feature_scores.csv"  # Path to the previously saved feature scores
feature_scores_df = pd.read_csv(feature_scores_path)

print(f"Loaded feature scores from: {feature_scores_path}")
print(f"Number of rows in feature_scores_df: {len(feature_scores_df)}")

# Take the top 500 features based on score
top_n = 500
selected_features = feature_scores_df.head(top_n)["Feature"].tolist()
selected_features_set = set(selected_features)
print(f"Selected top {top_n} features. Total unique selected features: {len(selected_features_set)}")

########################################################
# Step 2: Identify which of these top 500 appear in ALL
#         patient CSV files (common across all patients)
########################################################
print("Step 2: Identify common features across all patients")
root_dir = r"C:\Users\Pouya\Documents\university\Bachelor - AUT\Final Project\Data Set\CSVs"  # Same root folder from previous step

# We'll iteratively refine this set of features by intersecting
# with the columns present in each patient's CSV.
common_features = selected_features_set.copy()

# Also keep track of all data for final concatenation
all_dataframes = []

for patient_folder in os.listdir(root_dir):
    patient_path = os.path.join(root_dir, patient_folder)
    if os.path.isdir(patient_path):
        print(f"Processing folder: {patient_folder}")
        # Check all CSV files in this folder
        for file in os.listdir(patient_path):
            if file.endswith(".csv"):
                file_path = os.path.join(patient_path, file)
                print(f"  Reading CSV file: {file_path}")
                df = pd.read_csv(file_path)

                # Feature columns are everything except the first (assumed to be the name, e.g., 'Window_Name')
                # and the last (label)
                feature_cols = df.columns[1:-1]

                # Intersect with the existing common_features
                common_features = common_features.intersection(set(feature_cols))
                print(f"  Current number of common features: {len(common_features)}")

################################################
# Step 3: Build a single DataFrame by combining
#         only the common features + label and the name column
################################################
print("Step 3: Build a single DataFrame using the common features and name column")
# Convert the set of common features to a sorted list for consistency
common_features = sorted(list(common_features))
print(f"Final common features count: {len(common_features)}")

for patient_folder in os.listdir(root_dir):
    patient_path = os.path.join(root_dir, patient_folder)
    if os.path.isdir(patient_path):
        # For each CSV in this patient's folder
        for file in os.listdir(patient_path):
            if file.endswith(".csv"):
                file_path = os.path.join(patient_path, file)
                print(f"  Processing CSV file for final dataset: {file_path}")
                df = pd.read_csv(file_path)

                # The name column is assumed to be the first column
                name_col = df.columns[0]
                # The label column is assumed to be the last column
                label_col = df.columns[-1]
                # Get only the common feature columns that exist in the current DataFrame
                selected_cols = [col for col in common_features if col in df.columns]

                # Build a sub-dataframe with the name, common features, and label
                sub_df = df[[name_col] + selected_cols + [label_col]].copy()

                # Optionally, add patient ID info if needed
                # sub_df["PatientID"] = patient_folder

                all_dataframes.append(sub_df)
                print(f"  Sub-dataframe shape: {sub_df.shape}")

############################################
# Step 4: Concatenate and save the final CSV
############################################
print("Step 4: Concatenate all dataframes")
final_df = pd.concat(all_dataframes, ignore_index=True)
print(f"Final dataframe shape: {final_df.shape}")

output_path = r"C:\Users\Pouya\Documents\university\Bachelor - AUT\Final Project\Data Set\combined data\combined_data.csv"  # Path to save the combined data
os.makedirs(os.path.dirname(output_path), exist_ok=True)
final_df.to_csv(output_path, index=False)

print(f"Done! Final dataset saved to: {output_path}")
