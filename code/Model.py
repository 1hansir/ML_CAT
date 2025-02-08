import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class EncoderDecoderModel(nn.Module):
    def __init__(self, num_all_features, input_channels, hidden_dim):
        super(EncoderDecoderModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=hidden_dim, kernel_size=(3, 3), padding=(1, 1))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(3, 3), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, 1)  # Fully connected layer to predict a single value per feature
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary outputs

    def forward(self, x):
        """
        Forward pass for the model.
        """
        x = x.permute(0, 2, 1).unsqueeze(-1)  # Reshape for Conv2D
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x.squeeze(-1).permute(0, 2, 1)
        x = self.fc(x)
        x = self.sigmoid(x)  # Apply sigmoid to get binary probabilities
        return x.squeeze(-1)


class EncoderDecoderModel_confidence(nn.Module):
    def __init__(self, num_all_features, input_channels, hidden_dim):
        super(EncoderDecoderModel_confidence, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=hidden_dim, kernel_size=(3, 3), padding=(1, 1))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(3, 3), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.fc_result = nn.Linear(hidden_dim, 1)  # Fully connected layer for predicted result
        self.fc_confidence = nn.Linear(hidden_dim, 1)  # Fully connected layer for confidence score
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary predictions and confidence scores

    def forward(self, x):
        """
        Forward pass for the model.
        """
        x = x.permute(0, 2, 1).unsqueeze(-1)  # Reshape for Conv2D
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x.squeeze(-1).permute(0, 2, 1)

        # Separate branches for prediction and confidence score
        result = self.fc_result(x)  # Shape: (batch_size, num_features, 1)
        confidence = self.fc_confidence(x)  # Shape: (batch_size, num_features, 1)

        # Apply sigmoid to both branches
        result = self.sigmoid(result)  # Binary predictions (0 to 1)
        # Normalize confidence scores with softmax along the feature dimension
        confidence = nn.functional.softmax(confidence, dim=1)  # Ensures weights sum to 1 for each sample

        return torch.cat([result, confidence], dim=-1)  # Shape: (batch_size, num_features, 2)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_heads, num_layers, dropout=0.1):
        """
        Transformer-style encoder-decoder model for feature prediction.

        Parameters:
        - input_dim: Number of features (num_all_features).
        - embedding_dim: Dimensionality of feature embeddings.
        - num_heads: Number of attention heads in the multi-head attention mechanism.
        - num_layers: Number of layers in the encoder and decoder.
        - dropout: Dropout rate for regularization.
        """
        super(TransformerModel, self).__init__()

        # Input embedding layer
        self.embedding = nn.Linear(input_dim, embedding_dim)

        # Transformer encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads, dropout=dropout, dim_feedforward=4 * embedding_dim
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Transformer decoder
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim, nhead=num_heads, dropout=dropout, dim_feedforward=4 * embedding_dim
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        # Final output layer
        self.output_layer = nn.Linear(embedding_dim, 1)  # Predict scalar value for each feature

        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        """
        Forward pass through the model.

        Parameters:
        - X: Input tensor of shape (batch_size, num_features, input_dim).

        Returns:
        - predictions: Tensor of predicted values (batch_size, num_features).
        """
        batch_size, num_features, input_dim = X.shape

        # Flatten features and pass through embedding layer
        X_flat = X.view(-1, input_dim)  # Shape: (batch_size * num_features, input_dim)
        embeddings = self.embedding(X_flat)  # Shape: (batch_size * num_features, embedding_dim)
        embeddings = embeddings.view(batch_size, num_features, -1)  # Shape: (batch_size, num_features, embedding_dim)

        # Apply transformer encoder
        encoded = self.encoder(embeddings.permute(1, 0, 2))  # Shape: (num_features, batch_size, embedding_dim)

        # Apply transformer decoder
        decoded = self.decoder(encoded, encoded)  # Shape: (num_features, batch_size, embedding_dim)

        # Pass through the output layer
        predictions = self.sigmoid(self.output_layer(decoded.permute(1, 0, 2)))  # Shape: (batch_size, num_features, 1)
        predictions = predictions.squeeze(-1)  # Shape: (batch_size, num_features)

        return predictions

class TransformerModel_Confidence(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_heads=4, num_layers=3, dropout=0.1, batch_first=True):
        super().__init__()
        
        self.batch_first = batch_first
        self.embedding = nn.Linear(input_dim, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=4*embedding_dim,
            dropout=dropout,
            batch_first=batch_first,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(embedding_dim, 1)
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.output(x)  # (batch_size, seq_len, 1)
        return x.squeeze(-1)  # (batch_size, seq_len)


def weighted_bce_loss(predictions, targets, confidence_scores):
    """
    Compute the weighted binary cross-entropy loss.

    Parameters:
    - predictions: Predicted values (batch_size, num_features).
    - targets: Ground truth values (batch_size, num_features).
    - confidence_scores: Confidence scores (batch_size, num_features).

    Returns:
    - loss: Weighted BCE loss.
    """
    bce_loss = nn.BCELoss(reduction='none')  # Binary cross-entropy without reduction
    loss = bce_loss(predictions, targets)  # Compute BCE loss per element
    weighted_loss = (loss * confidence_scores).mean()  # Weight by confidence scores
    return weighted_loss


class LinearModel(nn.Module):
    def __init__(self, num_all_features, input_channels, hidden_dim):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(input_channels, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, 1)  # Output layer for predictions
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass for the model.
        
        Parameters:
        - x: Input tensor of shape (batch_size, num_features, input_channels)
        
        Returns:
        - predictions: Tensor of shape (batch_size, num_features)
        """
        # Process each feature independently
        batch_size, num_features, input_channels = x.shape
        x = x.view(-1, input_channels)  # Reshape to (batch_size * num_features, input_channels)
        
        # Pass through fully connected layers
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        
        # Reshape back to (batch_size, num_features)
        return x.view(batch_size, num_features)

class LinearModel_Confidence(nn.Module):
    def __init__(self, num_all_features, input_channels, hidden_dim):
        super(LinearModel_Confidence, self).__init__()
        self.fc1 = nn.Linear(input_channels, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        
        # Separate branches for prediction and confidence
        self.fc_result = nn.Linear(hidden_dim, 1)  # Output layer for predictions
        self.fc_confidence = nn.Linear(hidden_dim, 1)  # Output layer for confidence scores
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass for the model.
        
        Parameters:
        - x: Input tensor of shape (batch_size, num_features, input_channels)
        
        Returns:
        - output: Tensor of shape (batch_size, num_features, 2) containing predictions and confidence scores
        """
        # Process each feature independently
        batch_size, num_features, input_channels = x.shape
        x = x.view(-1, input_channels)  # Reshape to (batch_size * num_features, input_channels)
        
        # Shared layers
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        
        # Split into prediction and confidence branches
        result = self.sigmoid(self.fc_result(x))  # Binary predictions (0 to 1)
        confidence = self.fc_confidence(x)
        
        # Reshape outputs
        result = result.view(batch_size, num_features, 1)
        confidence = confidence.view(batch_size, num_features, 1)
        
        # Apply softmax to confidence scores along the feature dimension
        confidence = nn.functional.softmax(confidence, dim=1)  # Ensures weights sum to 1 for each sample
        
        return torch.cat([result, confidence], dim=-1)  # Shape: (batch_size, num_features, 2)

