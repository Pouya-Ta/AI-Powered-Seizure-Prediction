import os
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

print("Starting improved Naïve Bayes training and evaluation for binary classification (labels 1 and -1)...")

###############################
# Step 1: Load Train and Test Data
###############################
data_dir = r"C:\Users\Pouya\Documents\university\Bachelor - AUT\Final Project\Data Set\Splitted Dataset Binary"
train_path = os.path.join(data_dir, "trainData.csv")
test_path = os.path.join(data_dir, "testData.csv")

print(f"Loading train data from: {train_path}")
df_train = pd.read_csv(train_path)
print(f"Train dataset shape: {df_train.shape}")

print(f"Loading test data from: {test_path}")
df_test = pd.read_csv(test_path)
print(f"Test dataset shape: {df_test.shape}")

# Filter out rows with label 0 to ensure only labels -1 and 1 remain
label_col = df_train.columns[-1]
df_train = df_train[df_train[label_col] != 0]
df_test = df_test[df_test[label_col] != 0]
print("Unique training labels after filtering:", sorted(df_train[label_col].unique()))
print("Unique testing labels after filtering:", sorted(df_test[label_col].unique()))

####################################
# Step 2: Prepare Features and Labels
####################################
# Assume the first column contains record names (non-numeric)
name_col = df_train.columns[0]

# Drop the name and label columns to prepare features
X_train = df_train.drop(columns=[name_col, label_col])
y_train = df_train[label_col]

X_test = df_test.drop(columns=[name_col, label_col])
y_test = df_test[label_col]

print(f"Number of training samples: {X_train.shape[0]}")
print(f"Number of test samples: {X_test.shape[0]}")

###############################
# Step 3: Build Pipeline and Hyperparameter Tuning for Gaussian Naïve Bayes
###############################
# Create a pipeline that standardizes data and applies GaussianNB
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('nb', GaussianNB())
])

# Tune the var_smoothing parameter (which helps with numerical stability)
param_grid = {
    'nb__var_smoothing': [1e-09, 1e-08, 1e-07, 1e-06]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
print("Performing grid search for Naïve Bayes hyperparameter tuning...")
grid_search.fit(X_train, y_train)

print("Best parameters found:", grid_search.best_params_)
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

###############################
# Step 4: Evaluate the Best Model on Test Data
###############################
best_nb = grid_search.best_estimator_
y_pred = best_nb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Test set accuracy: {accuracy:.4f}")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Ictal (-1)", "Preictal (1)"]))
