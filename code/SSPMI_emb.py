import pandas as pd
import numpy as np
import re

# Load the dataset
file_path = "../data/updated_ML_data.csv"  # Replace with your local file path
data = pd.read_csv(file_path)

data = data.iloc[:100000]

# Select only columns with names matching X{i}
feature_columns = [col for col in data.columns if re.match(r'^X\d+$', col)]
print(feature_columns)


# Function to compute the co-occurrence matrix
def compute_cooccurrence_matrix(df, features):
    """
    Compute a co-occurrence matrix for features in the dataset.

    Parameters:
    - df: DataFrame containing the data.
    - features: List of feature names.

    Returns:
    - cooccurrence_matrix: Co-occurrence matrix (numpy array).
    """
    num_features = len(features)
    cooccurrence_matrix = np.zeros((num_features, num_features), dtype=np.float32)

    for _, row in df.iterrows():
        indices = [i for i, v in enumerate(row[features]) if v > 0]  # Features present in the sample
        for i in indices:
            for j in indices:
                if i != j:  # Exclude self-co-occurrence
                    cooccurrence_matrix[i, j] += 1

    print(cooccurrence_matrix)
    return cooccurrence_matrix


# Function to compute the SSPMI matrix
def compute_sspmi_matrix(cooccurrence_matrix, shift=1):
    """
    Compute the SSPMI (Shifted Positive Pointwise Mutual Information) matrix.

    Parameters:
    - cooccurrence_matrix: Co-occurrence matrix (numpy array).
    - shift: Shift value for SSPMI.

    Returns:
    - sspmi_matrix: SSPMI matrix (numpy array).
    """
    total_count = cooccurrence_matrix.sum()
    row_sums = cooccurrence_matrix.sum(axis=1)
    col_sums = cooccurrence_matrix.sum(axis=0)

    # Compute PPMI
    ppmi_matrix = np.maximum(
        np.log((cooccurrence_matrix * total_count + 1e-9 ) / (row_sums[:, None] * col_sums[None, :] + 1e-9)), 0
    )

    # Apply SSPMI shift
    sspmi_matrix = ppmi_matrix - shift
    sspmi_matrix[sspmi_matrix < 0] = 0  # Set negative values to 0

    return sspmi_matrix


# Function to factorize the SSPMI matrix
def factorize_sspmi_matrix(sspmi_matrix, embedding_dim=50):
    """
    Factorize the SSPMI matrix using SVD to generate feature embeddings.

    Parameters:
    - sspmi_matrix: SSPMI matrix (numpy array).
    - embedding_dim: Desired dimensionality of embeddings.

    Returns:
    - feature_embeddings: Embeddings for each feature (numpy array).
    """
    U, S, _ = np.linalg.svd(sspmi_matrix, full_matrices=False)
    embeddings = U[:, :embedding_dim] * S[:embedding_dim]
    return embeddings


# Calculate the co-occurrence matrix
cooccurrence_matrix = compute_cooccurrence_matrix(data, feature_columns)

# Compute the SSPMI matrix
sspmi_matrix = compute_sspmi_matrix(cooccurrence_matrix, shift=0.05)

# Factorize the SSPMI matrix to generate embeddings
embedding_dim = 50  # Dimensionality of embeddings
sspmi_embeddings = factorize_sspmi_matrix(sspmi_matrix, embedding_dim=embedding_dim)

# Save SSPMI embeddings to a CSV file
embedding_output_path = "../data/sspmi_feature_embeddings.csv"  # Replace with your desired output path
sspmi_embeddings_df = pd.DataFrame(sspmi_embeddings, index=feature_columns)
sspmi_embeddings_df.to_csv(embedding_output_path, header=False)
print(f"SSPMI embeddings saved to {embedding_output_path}")

