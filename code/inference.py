import torch
import pandas as pd
import numpy as np
from Model import EncoderDecoderModel, EncoderDecoderModel_confidence, TransformerModel, TransformerModel_Confidence  # Import the model class from Model.py


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model_from_checkpoint(checkpoint_path, num_all_features, input_channels, hidden_dim, Confidence_model, Transformer_model):
    """
    Load the model and optimizer states from a checkpoint file.

    Parameters:
    - checkpoint_path: Path to the checkpoint file.
    - num_all_features: Number of features in the dataset.
    - input_channels: Number of input channels for the model.
    - hidden_dim: Number of hidden dimensions in the model.

    Returns:
    - model: The trained model loaded from the checkpoint.
    """
    if not Transformer_model:
        if Confidence_model:
            model = EncoderDecoderModel_confidence(num_all_features, input_channels, hidden_dim)
        else:
            model = EncoderDecoderModel(num_all_features, input_channels, hidden_dim)

    elif Transformer_model:
        INPUT_DIM = input_channels  # Indicator, feature value, embedding
        EMBEDDING_DIM = 64  # Transformer embedding dimension
        NUM_HEADS = 4  # Number of attention heads
        NUM_LAYERS = 3  # Number of transformer layers
        DROPOUT = 0.1

        if Confidence_model:
            model = TransformerModel_Confidence(
                input_dim=INPUT_DIM,
                embedding_dim=EMBEDDING_DIM,
                num_heads=NUM_HEADS,
                num_layers=NUM_LAYERS,
                dropout=DROPOUT
            ).to(device)
        else:
            model = TransformerModel(
                input_dim=INPUT_DIM,
                embedding_dim=EMBEDDING_DIM,
                num_heads=NUM_HEADS,
                num_layers=NUM_LAYERS,
                dropout=DROPOUT
            ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model



def preprocess_data(file_path, item_parameters, all_features, sspmi_embedding_file = None):
    """
    Preprocess the inference dataset for model input.

    Parameters:
    - file_path: Path to the CSV file containing the test data.
    - item_parameters: Dictionary mapping feature names to embeddings.

    Returns:
    - input_tensor: Preprocessed tensor for model input.
    - original_data: Original DataFrame for reconstructing output.
    """
    data = pd.read_csv(file_path, index_col=0)
    num_samples = data.shape[0]

    # Convert NA to zero and create indicator
    indicator = (~data.isna()).astype(int).values  # 1 for known, 0 for unknown
    values = data.fillna(0).values  # Replace NA with 0 for processing

    if sspmi_embedding_file:
        # Load SSPMI embeddings from file
        all_features_sspmi = [f"X{i}" for i in range(1, len(all_features) + 1)]  # Total 44 features
        sspmi_embeddings_df = pd.read_csv(sspmi_embedding_file, header=None, index_col=0)
        sspmi_embeddings = sspmi_embeddings_df.loc[all_features_sspmi].values

        # Combine old embeddings (from item_parameters) with SSPMI embeddings
        concatenated_embeddings = []
        for i, feature in enumerate(all_features):
            old_embedding = np.array(item_parameters[feature])  # Old embedding (from column 'b')
            sspmi_embedding = sspmi_embeddings[i]  # SSPMI embedding
            concatenated_embeddings.append(np.hstack([old_embedding, sspmi_embedding]))

        # Create embeddings tensor for all features
        embeddings = np.array(concatenated_embeddings)  # Shape: (num_all_features, embedding_dim)
        embeddings = np.tile(embeddings, (num_samples, 1, 1))  # Shape: (num_samples, num_all_features, embedding_dim)

    else:
        # Prepare embeddings for each feature
        embeddings = np.array([item_parameters[col] for col in all_features])
        embeddings = np.tile(embeddings, (data.shape[0], 1))  # Repeat embeddings for each sample

    # Combine indicator, values, and embeddings into a single input tensor
    input_tensor = np.concatenate([indicator[:, :, None], values[:, :, None], embeddings], axis=-1)

    return torch.tensor(input_tensor, dtype=torch.float32), data


def predict_and_fill(input_tensor, model, original_data, Confidence_model):
    """
    Predict missing values and fill them into the original dataset.
    """
    with torch.no_grad():
        if Confidence_model:
            predictions = model(input_tensor)[:, :, 0]
        else:
            predictions = model(input_tensor)
        binary_predictions = (predictions > 0.5).float()  # Threshold predictions to 0 or 1

    # Fill missing values in the original DataFrame
    filled_data = original_data.copy()
    indicator = input_tensor[:, :, 0].numpy()  # Extract indicator
    mask = (indicator == 0)  # Missing values mask
    filled_data.values[mask] = binary_predictions.numpy()[mask]  # Replace only missing values

    return filled_data


def main():
    # Paths to files
    shuffle_known_feature = False
    Confidence_model = False
    Transformer_model = False
    Emb = 'SSPMI'  # Emb = ['Naive','SPPMI']
    N_KNOWN_FEATURE = 36

    checkpoint_path = f'../checkpoints/best_model_S{shuffle_known_feature}_C{Confidence_model}_T{Transformer_model}_{Emb}_F{N_KNOWN_FEATURE}.pth'
    test_file_path = f'../data/2b_Inferenced/Resp_Validation_{N_KNOWN_FEATURE}.csv'
    output_file_path = f'../data/inferenced/Inferenced_{N_KNOWN_FEATURE}.csv'

    # Model parameters
    num_all_features = 44  # Adjust based on your dataset
    if Emb == 'Naive':
        input_channels = 3
        sspmi_file = None
    elif Emb == 'SSPMI':
        input_channels = 47
        sspmi_file = "../data/sspmi_feature_embeddings.csv"

    hidden_dim = 64

    # Load item parameters (replace with actual parameters)
    item_parameters = {f'{i}': 0.1 for i in range(1, num_all_features+1)}  # Example item parameters

    # Load the trained model
    model = load_model_from_checkpoint(checkpoint_path, num_all_features, input_channels, hidden_dim, Confidence_model, Transformer_model)

    # Preprocess the test data
    all_features = [f"{i}" for i in range(1, num_all_features + 1)]  # Total 44 features
    input_tensor, original_data = preprocess_data(test_file_path, item_parameters, all_features, sspmi_file)

    # Perform inference
    filled_data = predict_and_fill(input_tensor, model, original_data, Confidence_model)

    # Save the filled data to a new file
    filled_data.to_csv(output_file_path, index=True)
    print(f"Predicted results saved to {output_file_path}")


if __name__ == "__main__":
    main()



