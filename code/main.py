import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import Preprocessing
from Model import EncoderDecoderModel, EncoderDecoderModel_confidence, TransformerModel, TransformerModel_Confidence
from Model import weighted_bce_loss
from Model import LinearModel, LinearModel_Confidence
from sklearn.metrics import roc_auc_score
import numpy as np
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Configurations
TRAIN_FILE = "../data/updated_ML_data.csv"
VALID_FILE = "../data/updated_validation_data.csv"
TRAIN_PARAMS_FILE = "../data/parameters.csv"
VALID_PARAMS_FILE = "../data/parameters_validation.csv"
sspmi_embedding_file = "../data/sspmi_feature_embeddings.csv"
N_ALL_FEATURES = 44
ALL_FEATURES = [f"X{i}" for i in range(1, N_ALL_FEATURES+1)]  # Total 44 features

shuffle_known_feature = False
Confidence_model = False
Transformer_model = False
Linear_model = True  # Set this to True to use linear models instead of CNN or Transformer
Emb = 'SSPMI'   # Emb = ['Naive','SSPMI']


N_KNOWN_FEATURES_ = [30, 35, 40]
TRAIN_SAMPLE_NUM = 100000 # -1 = ALL
SHUFFLE_NUM = 2
BATCH_SIZE = 32
HIDDEN_DIM = 64
EPOCHS = 20
LEARNING_RATE = 0.001
RANDOM_SEED = 42  # For reproducibility

for N_KNOWN_FEATURES in N_KNOWN_FEATURES_:
    # Load datasets and parameters
    print("Loading datasets and parameters for Known Features # == {}".format(N_KNOWN_FEATURES))
    train_df, val_df = Preprocessing.load_dataset(TRAIN_FILE, VALID_FILE)
    train_params = Preprocessing.load_item_parameters(TRAIN_PARAMS_FILE, N_ALL_FEATURES)
    val_params = Preprocessing.load_item_parameters(VALID_PARAMS_FILE, N_ALL_FEATURES)


    # Limit training set to the first 10,000 samples
    train_df = train_df.iloc[:TRAIN_SAMPLE_NUM]


    # Preprocess training and validation data
    print("Preprocessing training and validation data...")
    if Emb == 'Naive':
        X_train, Y_train = Preprocessing.preprocess_data_naiveemb(
            train_df, ALL_FEATURES, num_known_features=N_KNOWN_FEATURES,
            item_parameters=train_params,
            shuffle_known_feature=shuffle_known_feature,
            shuffle_num=SHUFFLE_NUM
        )
        X_val, Y_val = Preprocessing.preprocess_data_naiveemb(
            val_df, ALL_FEATURES, num_known_features=N_KNOWN_FEATURES,
            item_parameters=val_params,
            random_seed=RANDOM_SEED * 2,
            shuffle_known_feature=shuffle_known_feature,
            shuffle_num=SHUFFLE_NUM
        )

    elif Emb == 'SSPMI':
        X_train, Y_train = Preprocessing.preprocess_data(
            train_df, ALL_FEATURES, num_known_features=N_KNOWN_FEATURES,
            item_parameters=train_params,
            sspmi_embedding_file=sspmi_embedding_file,
            shuffle_known_feature = shuffle_known_feature,
            shuffle_num=SHUFFLE_NUM
        )
        X_val, Y_val = Preprocessing.preprocess_data(
            val_df, ALL_FEATURES, num_known_features=N_KNOWN_FEATURES,
            item_parameters=val_params,
            random_seed=RANDOM_SEED*2,
            sspmi_embedding_file=sspmi_embedding_file,
            shuffle_known_feature = shuffle_known_feature,
            shuffle_num=SHUFFLE_NUM
        )

    # Convert to PyTorch tensors
    X_train_tensor = X_train  # Shape: (10000, num_all_features, 3)
    Y_train_tensor = Y_train  # Shape: (10000, num_all_features)
    X_val_tensor = X_val  # Shape: (num_samples_val, num_all_features, 3)
    Y_val_tensor = Y_val  # Shape: (num_samples_val, num_all_features)


    # Create data loaders
    print("Creating data loaders...")
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model Parameters
    num_all_features = len(ALL_FEATURES)
    if Emb == 'Naive':
        input_channels = 3 # ['indicator', 'value', 'param']
    elif Emb == 'SSPMI':
        input_channels = 3 + 44 # ['indicator', 'value', 'param', 44 SSPMI emb]



    # Initialize the model
    if Linear_model:
        if Confidence_model:
            model = LinearModel_Confidence(
                num_all_features=num_all_features,
                input_channels=input_channels,
                hidden_dim=HIDDEN_DIM
            )
        else:
            model = LinearModel(
                num_all_features=num_all_features,
                input_channels=input_channels,
                hidden_dim=HIDDEN_DIM
            )
    elif Confidence_model:
        model = EncoderDecoderModel_confidence(
            num_all_features=num_all_features,
            input_channels=input_channels,
            hidden_dim=HIDDEN_DIM
        )
    else:
        model = EncoderDecoderModel(
            num_all_features=num_all_features,
            input_channels=input_channels,
            hidden_dim=HIDDEN_DIM
        )

    if Transformer_model:
        # Model parameters
        NUM_FEATURES = len(ALL_FEATURES)  # Total number of features
        INPUT_DIM = input_channels  # Indicator, feature value, embedding
        EMBEDDING_DIM = 64  # Transformer embedding dimension
        NUM_HEADS = 4  # Number of attention heads
        NUM_LAYERS = 3  # Number of transformer layers
        DROPOUT = 0.1

        # Initialize the transformer model
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

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Directory to save checkpoints
    CHECKPOINT_DIR = "../checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Initialize variables to track the best validation performance
    best_val_auc = 0.0  # Best validation AUC so far

    # Training and validation pipeline
    print("Starting training...")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        # Training phase
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []
        with tqdm(total=len(train_loader), desc="Training", unit="batch") as pbar:
            for X_batch, Y_batch in train_loader:
                optimizer.zero_grad()
                # Forward pass
                output = model(X_batch)  # Shape: (batch_size, num_features, 2)
                if Confidence_model:
                    predictions = output[:, :, 0]  # Predicted results
                    confidence_scores = output[:, :, 1]  # Confidence scores

                    # Compute weighted loss
                    loss = weighted_bce_loss(predictions, Y_batch, confidence_scores)
                else:
                    predictions = output
                    loss = criterion(predictions, Y_batch)

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                # Store predictions and true values for AUC calculation
                train_preds.append(predictions.detach().cpu().numpy())
                train_targets.append(Y_batch.cpu().numpy())

                # Update progress bar
                pbar.update(1)

        # Calculate training AUC
        train_preds = np.concatenate(train_preds, axis=0).flatten()  # Flatten predictions
        train_targets = np.concatenate(train_targets, axis=0).flatten()  # Flatten targets
        train_auc = roc_auc_score(train_targets > 0, train_preds)  # Binary classification for AUC

        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_bin_preds = []
        val_targets = []
        with tqdm(total=len(val_loader), desc="Validation", unit="batch") as pbar:
            with torch.no_grad():
                for X_batch, Y_batch in val_loader:
                    # Forward pass
                    output = model(X_batch)

                    if Confidence_model:
                        predictions = output[:, :, 0]
                        confidence_scores = output[:, :, 1]

                        # Compute validation loss
                        binary_predictions = (predictions > 0.5).float()
                        loss = weighted_bce_loss(binary_predictions, Y_batch, confidence_scores)
                    else:
                        predictions = output
                        binary_predictions = (predictions > 0.5).float()
                        loss = criterion(binary_predictions, Y_batch)

                    val_loss += loss.item()

                    # Store predictions and true values for AUC calculation
                    val_preds.append(predictions.cpu().numpy())
                    val_bin_preds.append(binary_predictions.cpu().numpy())
                    val_targets.append(Y_batch.cpu().numpy())

                    # Update progress bar
                    pbar.update(1)

        # Calculate validation AUC
        val_preds = np.concatenate(val_preds, axis=0).flatten()  # Flatten predictions
        val_bin_preds = np.concatenate(val_bin_preds, axis=0).flatten()  # Flatten predictions
        val_targets = np.concatenate(val_targets, axis=0).flatten()  # Flatten targets
        val_auc = roc_auc_score(val_targets > 0, val_preds)  # Binary classification for AUC
        val_bin_auc = roc_auc_score(val_targets > 0, val_bin_preds)  # Binary classification for AUC

        # Print metrics
        print(f"Train Loss: {train_loss:.8f}, Train AUC: {train_auc:.9f}, "
              f"Val Loss: {val_loss:.8f}, Val AUC: {val_auc:.9f}"
              f"Val Binary Loss: {val_loss:.8f}, Val Binary AUC: {val_bin_auc:.9f}")

        # Save checkpoint if validation AUC improves
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_model_S{shuffle_known_feature}_C{Confidence_model}_T{Transformer_model}_{Emb}_F{N_KNOWN_FEATURES}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_auc': val_auc
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path} (Val Binary AUC: {val_bin_auc:.9f})")

