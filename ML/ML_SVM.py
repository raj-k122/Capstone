import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

def process_excel_file(file_path):
    """Loads and combines landmark sheets from a single Excel file,
       reading only the 2nd and 3rd columns, and renaming columns."""
    all_sheets = pd.read_excel(file_path, sheet_name=None)
    combined_data = pd.DataFrame()
    for sheet_name, df in all_sheets.items():
        # Select only the 2nd and 3rd columns (index 1 and 2)
        selected_columns = df.iloc[:, 1:3]
        # Rename columns to ensure uniqueness
        selected_columns.columns = [f"{sheet_name}_dx", f"{sheet_name}_dy"]
        combined_data = pd.concat([combined_data, selected_columns], axis=1)
    return combined_data

def load_and_prepare_data(healthy_files, stroke_files):
    """Loads, processes, and combines data from all Excel files."""
    all_data = []
    labels = []
    subject_row_counts = [] #store the row counts for each subject.

    # Process healthy files
    original_dir = os.getcwd()  # Store original directory
    os.chdir("../generated/healthy/") # Change Directory
    for file_name in healthy_files:
        data = process_excel_file(file_name)
        all_data.append(data)
        labels.append(0)  # 0 for healthy
        subject_row_counts.append(len(data))
    os.chdir(original_dir) # Return to original Directory

    # Process stroke files
    original_dir = os.getcwd()  # Store original directory
    os.chdir("../generated/patients/p1")
    for file_path in stroke_files:
        data = process_excel_file(file_path)
        all_data.append(data)
        labels.append(1)  # 1 for stroke
        
        subject_row_counts.append(len(data))

    combined_df = pd.concat(all_data, ignore_index=True)
    # Correctly assign labels
    repeated_labels = []
    label_index = 0
    for count in subject_row_counts:
        repeated_labels.extend([labels[label_index]] * count)
        label_index += 1

    combined_df["label"] = repeated_labels
    print(combined_df)
    return combined_df



# Example usage (replace with your file paths)
healthy_files = ["forward #1.xlsx", "forward #2.xlsx", "forward #3.xlsx", "forward #4.xlsx", "forward #5.xlsx"]
stroke_files = ["front_1_p1.xlsx", "front_2_p1.xlsx"]

all_data = load_and_prepare_data(healthy_files, stroke_files)


# Create feature matrix and target vector
# X = all_data.drop("label", axis=1)
# y = all_data["label"]

# # # Train-test split, scaling, SVM training, and evaluation
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# model = SVC(kernel="rbf")
# model.fit(X_train_scaled, y_train)

# y_pred = model.predict(X_test_scaled)

# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))