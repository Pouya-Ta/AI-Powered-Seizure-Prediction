import os
import pandas as pd

def label_files_in_folders(root_folder):
    for patient_folder in os.listdir(root_folder):
        patient_path = os.path.join(root_folder, patient_folder)
        if os.path.isdir(patient_path):  # Ensure it's a subfolder
            for file in os.listdir(patient_path):
                if file.endswith(".csv"):  # Process only CSV files
                    file_path = os.path.join(patient_path, file)
                    
                    # Read the CSV
                    df = pd.read_csv(file_path)

                    # Get the name of the first column. For safety, let's just do:
                    first_col_name = df.columns[0]

                    # Function to assign label based on the content of the cell
                    def assign_label(value):
                        # Ensure it's a string for "in" checks
                        value_str = str(value).lower()
                        if "preictal" in value_str:
                            return 1
                        elif "interictal" in value_str:
                            return 0
                        elif "ictal" in value_str:
                            return -1
                        else:
                            return None  # or any default you want

                    # Apply that function to every row in the first column
                    df["labels"] = df[first_col_name].apply(assign_label)
                    
                    # Save the new CSV (you can overwrite or save as new file)
                    labeled_filename = file.replace(".csv", "_labeled.csv")
                    labeled_file_path = os.path.join(patient_path, labeled_filename)
                    df.to_csv(labeled_file_path, index=False)
                    
                    print(f"Processed and labeled: {labeled_file_path}")

# Usage
root_directory = r"C:\Users\Pouya\Documents\university\Bachelor - AUT\Final Project\Data Set\CSVs"
label_files_in_folders(root_directory)