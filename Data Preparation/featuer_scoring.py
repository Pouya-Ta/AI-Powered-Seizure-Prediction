import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif, mutual_info_classif

print("Starting feature scoring script...")

# Define the root directory containing the 24 patient folders
root_dir = r"C:\Users\Pouya\Documents\university\Bachelor - AUT\Final Project\Data Set\CSVs"  # Change this to your actual path
output_path = r"C:\Users\Pouya\Documents\university\Bachelor - AUT\Final Project\Data Set\Selected_Features\feature_scores.csv"  # New path for saving the results
print(f"Root directory is set to: {root_dir}")
print(f"Output path is set to: {output_path}")

# Dictionary to store feature scores
feature_scores = {}

# Iterate over patient folders
print("Iterating over patient folders...")
for patient_folder in os.listdir(root_dir):
    patient_path = os.path.join(root_dir, patient_folder)
    if os.path.isdir(patient_path):
        print(f"  Found patient folder: {patient_folder}")

        # Iterate over all CSV files in the patient's folder
        for file in os.listdir(patient_path):
            if file.endswith(".csv"):
                file_path = os.path.join(patient_path, file)
                print(f"    Reading CSV file: {file_path}")
                df = pd.read_csv(file_path)
                print(f"    CSV file shape: {df.shape}")

                # Extract feature columns (excluding 'Window_Name' and label column)
                feature_columns = df.columns[1:-1]  # Assuming last column is the label
                labels = df.iloc[:, -1]  # Extract labels (-1, 0, 1)

                # Convert feature columns to numeric, coercing errors to NaN
                df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors='coerce')

                # Fill NaNs with the mean of each column
                df[feature_columns] = df[feature_columns].fillna(df[feature_columns].mean())

                # Fill any remaining NaNs (e.g., columns entirely NaN) with 0
                df[feature_columns] = df[feature_columns].fillna(0)

                # Debug: Check if any NaNs remain
                if df[feature_columns].isnull().any().any():
                    print("Warning: NaN values still exist in the following columns:")
                    print(df[feature_columns].columns[df[feature_columns].isnull().any()])
                else:
                    print("No NaN values detected in feature columns.")

                # Now compute f_classif
                f_values, p_values = f_classif(df[feature_columns], labels)


                # Store scores
                for idx, feature in enumerate(feature_columns):
                    if feature not in feature_scores:
                        feature_scores[feature] = []
                    feature_scores[feature].append(f_values[idx])
                print("    Feature scores updated.")

# Compute average score per feature across all patients
print("Computing average score per feature...")
final_feature_scores = {
    feature: np.mean(scores) for feature, scores in feature_scores.items()
}
print("Average score computation completed.")

# Convert to DataFrame and save as CSV
print("Converting scores to DataFrame...")
feature_score_df = pd.DataFrame(
    list(final_feature_scores.items()), columns=["Feature", "Score"]
)
feature_score_df = feature_score_df.sort_values(
    by="Score", ascending=False
)  # Sort by importance
print(f"DataFrame shape (features x 2): {feature_score_df.shape}")

# Ensure output directory exists
print("Ensuring output directory exists...")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

print(f"Saving DataFrame to CSV: {output_path}")
feature_score_df.to_csv(output_path, index=False)

print(f"Feature scoring completed. Scores saved in '{output_path}'.")
