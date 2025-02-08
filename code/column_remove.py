import pandas as pd
import re

# Re-import the uploaded datasets
train_data_path = '../data/ML_data.csv'
val_data_path = '../data/validation_data.csv'

train_data = pd.read_csv(train_data_path)
val_data = pd.read_csv(val_data_path)


# Function to remove 'X14' and rename subsequent columns
def process_data(data, column_to_remove='X14'):
    if column_to_remove in data.columns:
        # Drop the specified column
        data = data.drop(columns=[column_to_remove])

        # Rename columns X{i} to X{i-1} for columns after 'X14'
        columns = data.columns.tolist()
        updated_columns = []
        for col in columns:
            if col.startswith('X') and re.match(r'X\d+', col):  # Ensure column name matches 'X{i}'
                index = int(col[1:])
                updated_columns.append(f'X{index - 1}' if index > 14 else col)
            else:
                updated_columns.append(col)  # Keep non-matching columns unchanged
        data.columns = updated_columns
    else:
        print(f"Column '{column_to_remove}' not found in the dataset.")
    return data


# Process the training and validation datasets
updated_train_data = process_data(train_data)
updated_val_data = process_data(val_data)

# Save the updated datasets to new files
updated_train_data_path = '../data/updated_ML_data.csv'
updated_val_data_path = '../data/updated_validation_data.csv'

updated_train_data.to_csv(updated_train_data_path, index=False)
updated_val_data.to_csv(updated_val_data_path, index=False)


