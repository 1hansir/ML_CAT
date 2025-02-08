import pandas as pd
import numpy as np
import torch
from sklearn.utils import shuffle


def load_dataset(training_file, validation_file):
    """Load training and validation datasets."""
    train_df = pd.read_csv(training_file)
    val_df = pd.read_csv(validation_file)
    return train_df, val_df


def load_item_parameters(params_file, N_all_features):
    """
    Load initial parameters for features (items) in the dataset.

    Parameters:
    - params_file: Path to the CSV file with parameters (45 rows, multiple columns).

    Returns:
    - params_dict: Dictionary mapping feature names to their single-dimensional embeddings ('b' column).
    """
    params_df = pd.read_csv(params_file)
    feature_names = [f"X{i}" for i in range(1, N_all_features + 1)]  # Generate feature names (X1 to X45)
    params_df["item"] = feature_names  # Add feature names as identifiers
    params_dict = params_df.set_index("item")["b"].to_dict()  # Only use column 'b'
    return params_dict


def load_embeddings(embedding_file):
    """
    Load feature embeddings from a CSV file.

    Parameters:
    - embedding_file: Path to the embedding file.

    Returns:
    - embeddings: Dictionary mapping feature names to embeddings.
    """
    embedding_df = pd.read_csv(embedding_file, header=None, index_col=0)
    embeddings = {index: embedding_df.loc[index].values for index in embedding_df.index}
    return embeddings

def split_features(all_features, num_known_features, random_seed=42):
    """
    Randomly split features into known and unknown sets.

    Parameters:
    - all_features: List of all feature names.
    - num_known_features: Number of features to select as known.
    - random_seed: Seed for reproducibility.

    Returns:
    - known_features: List of known feature names.
    - unknown_features: List of unknown feature names.
    """
    np.random.seed(random_seed)
    shuffled_features = np.random.permutation(all_features)
    known_features = shuffled_features[:num_known_features]
    unknown_features = shuffled_features[num_known_features:]
    return list(known_features), list(unknown_features)

def preprocess_data(
    df, all_features, num_known_features, item_parameters, sspmi_embedding_file, shuffle_known_feature, shuffle_num, random_seed=42
):
    """
    Prepare data for training by creating a 3D tensor of feature indicators, values, and concatenated embeddings.

    Parameters:
    - df: DataFrame of the dataset.
    - all_features: List of all column names.
    - num_known_features: Number of features to assign as known for each sample.
    - item_parameters: Dict mapping column names to their embeddings from the 'b' column.
    - sspmi_embedding_file: Path to the SSPMI embeddings file.
    - shuffle_known_feature: Boolean, whether to shuffle the known features dynamically.
    - shuffle_num: Number of times to shuffle each sample.
    - random_seed: Seed for reproducibility.

    Returns:
    - X_tensor: A 3D tensor of shape (augmented_samples, num_all_features, 3).
    - Y_tensor: A 2D tensor of shape (augmented_samples, num_all_features).
    """
    np.random.seed(random_seed)
    num_samples = df.shape[0]
    all_feature_values = df[all_features].values  # Shape: (num_samples, num_all_features)

    # Load SSPMI embeddings from file
    sspmi_embeddings_df = pd.read_csv(sspmi_embedding_file, header=None, index_col=0)
    sspmi_embeddings = sspmi_embeddings_df.loc[all_features].values

    # Combine old embeddings (from item_parameters) with SSPMI embeddings
    concatenated_embeddings = []
    for i, feature in enumerate(all_features):
        old_embedding = np.array(item_parameters[feature])  # Old embedding (from column 'b')
        sspmi_embedding = sspmi_embeddings[i]  # SSPMI embedding
        concatenated_embeddings.append(np.hstack([old_embedding, sspmi_embedding]))

    # Create embeddings tensor for all features
    embeddings = np.array(concatenated_embeddings)  # Shape: (num_all_features, embedding_dim)
    embeddings = np.tile(embeddings, (num_samples, 1, 1))  # Shape: (num_samples, num_all_features, embedding_dim)

    # Initialize augmented data storage
    augmented_indicators = []
    augmented_values = []
    augmented_embeddings = []
    augmented_targets = []

    for i in range(num_samples):
        for _ in range(shuffle_num):
            # Update random seed for reproducibility
            np.random.seed(random_seed)
            random_seed += 1

            # Adjust number of known features dynamically if enabled
            if shuffle_known_feature:
                current_num_known_features = np.random.randint(low=1, high=len(all_features))
            else:
                current_num_known_features = num_known_features

            # Randomly select known features for the current shuffle
            known_features = np.random.choice(all_features, size=current_num_known_features, replace=False)
            known_indices = [all_features.index(f) for f in known_features]

            # Create indicator and values for the current shuffle
            indicator = np.zeros(len(all_features), dtype=float)
            feature_values = np.zeros(len(all_features), dtype=float)
            indicator[known_indices] = 1.0
            feature_values[known_indices] = all_feature_values[i, known_indices]

            # Zero out embeddings for unknown features
            current_embeddings = embeddings[i] * indicator[:, None]  # Shape: (num_all_features, embedding_dim)

            # Append the shuffled sample to augmented data
            augmented_indicators.append(indicator)
            augmented_values.append(feature_values)
            augmented_embeddings.append(current_embeddings)
            augmented_targets.append(all_feature_values[i])  # Original targets remain the same

    # Combine all augmented samples
    augmented_indicators = np.array(augmented_indicators)
    augmented_values = np.array(augmented_values)
    augmented_embeddings = np.array(augmented_embeddings)
    augmented_targets = np.array(augmented_targets)

    # Combine indicator, values, and embeddings into a single tensor
    X_tensor = np.concatenate([augmented_indicators[:, :, None], augmented_values[:, :, None], augmented_embeddings], axis= -1)
    Y_tensor = augmented_targets

    return torch.tensor(X_tensor, dtype=torch.float32), torch.tensor(Y_tensor, dtype=torch.float32)


def preprocess_data_naiveemb(
    df, all_features, num_known_features, item_parameters, shuffle_known_feature, shuffle_num, random_seed=42
):
    """
    Prepare data for training by creating a 3D tensor of feature indicators, values, and embeddings.
    Each sample is augmented by randomly shuffling known features multiple times.

    Parameters:
    - df: DataFrame of the dataset.
    - all_features: List of all column names.
    - num_known_features: Number of features to assign as known for each sample.
    - item_parameters: Dict mapping column names to their embeddings.
    - shuffle_known_feature: Boolean, whether to shuffle the known features dynamically.
    - shuffle_num: Number of times to shuffle each sample.
    - random_seed: Seed for reproducibility.

    Returns:
    - X_tensor: A 3D tensor of shape (augmented_samples, num_all_features, 3).
    - Y_tensor: A 2D tensor of shape (augmented_samples, num_all_features).
    """
    np.random.seed(random_seed)
    num_samples = df.shape[0]
    all_feature_values = df[all_features].values  # Shape: (num_samples, num_all_features)

    # Create embeddings tensor
    embeddings = np.array([item_parameters[feature] for feature in all_features])  # Shape: (num_all_features,)
    embeddings = np.tile(embeddings, (num_samples, 1))  # Shape: (num_samples, num_all_features)

    # Initialize augmented data storage
    augmented_indicators = []
    augmented_values = []
    augmented_embeddings = []
    augmented_targets = []

    for i in range(num_samples):
        for _ in range(shuffle_num):
            # Update random seed for reproducibility
            np.random.seed(random_seed)
            random_seed += 1

            # Adjust number of known features dynamically if enabled
            if shuffle_known_feature:
                current_num_known_features = np.random.randint(low=1, high=len(all_features) // 2)
            else:
                current_num_known_features = num_known_features

            # Randomly select known features for the current shuffle
            known_features = np.random.choice(all_features, size=current_num_known_features, replace=False)
            known_indices = [all_features.index(f) for f in known_features]

            # Create indicator and values for the current shuffle
            indicator = np.zeros_like(all_feature_values[i], dtype=float)
            feature_values = np.zeros_like(all_feature_values[i], dtype=float)

            indicator[known_indices] = 1.0
            feature_values[known_indices] = all_feature_values[i, known_indices]

            # Zero out embeddings for unknown features
            current_embeddings = embeddings[i] * indicator

            # Clone or detach before appending to avoid computational graph issues
            augmented_indicators.append(torch.tensor(indicator, dtype=torch.float32).clone())
            augmented_values.append(torch.tensor(feature_values, dtype=torch.float32).clone())
            augmented_embeddings.append(torch.tensor(current_embeddings, dtype=torch.float32).clone())
            augmented_targets.append(torch.tensor(all_feature_values[i], dtype=torch.float32).clone())

    # Combine all augmented samples
    augmented_indicators = torch.stack(augmented_indicators, dim=0)
    augmented_values = torch.stack(augmented_values, dim=0)
    augmented_embeddings = torch.stack(augmented_embeddings, dim=0)
    augmented_targets = torch.stack(augmented_targets, dim=0)

    # Combine indicator, values, and embeddings into a single tensor
    X_tensor = torch.stack([augmented_indicators, augmented_values, augmented_embeddings], dim=-1)
    Y_tensor = augmented_targets

    return X_tensor, Y_tensor

