import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import json
from datetime import datetime
from sklearn.metrics import roc_auc_score
import time
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

import Preprocessing
from Model import (
    EncoderDecoderModel_confidence,
    TransformerModel_Confidence,
    LinearModel_Confidence
)

# Configuration
TRAIN_FILE = "../data/updated_ML_data.csv"
VALID_FILE = "../data/updated_validation_data.csv"
TRAIN_PARAMS_FILE = "../data/parameters.csv"
VALID_PARAMS_FILE = "../data/parameters_validation.csv"
SSPMI_EMBEDDING_FILE = "../data/sspmi_feature_embeddings.csv"
RESULTS_DIR = "../results"
CHECKPOINT_DIR = "../checkpoints"
INFERENCE_DIR = "../data/inferenced"

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(INFERENCE_DIR, exist_ok=True)

# Common parameters
N_ALL_FEATURES = 44
ALL_FEATURES = [f"X{i}" for i in range(1, N_ALL_FEATURES+1)]
N_KNOWN_FEATURES = 35  # Fixed to 35
TRAIN_SAMPLE_NUM = 50000
SHUFFLE_NUM = 2
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
RANDOM_SEED = 42

class FocalLoss(nn.Module):
    def __init__(self, alpha=2.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets, class_weights=None):
        # Ensure inputs and targets are 2D
        if inputs.dim() > 2:
            inputs = inputs.squeeze(-1)
        if targets.dim() > 2:
            targets = targets.squeeze(-1)
            
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_term = self.alpha * (1-pt)**self.gamma
        
        if class_weights is not None:
            w0 = class_weights[0].to(inputs.device)
            w1 = class_weights[1].to(inputs.device)
            weights = torch.where(targets == 1, w1, w0)
            return (weights * focal_term * bce_loss).mean()
        
        return (focal_term * bce_loss).mean()



def train_model(model_config):
    """Train model with both Focal Loss and class weights."""
    # Unpack configuration
    hidden_dim = model_config['hidden_dim']
    model_type = model_config['model_type']
    embedding_type = model_config['embedding_type']
    
    # Load and preprocess data
    train_df, val_df = Preprocessing.load_dataset(TRAIN_FILE, VALID_FILE)
    train_params = Preprocessing.load_item_parameters(TRAIN_PARAMS_FILE, N_ALL_FEATURES)
    val_params = Preprocessing.load_item_parameters(VALID_PARAMS_FILE, N_ALL_FEATURES)
    
    # Limit training set to first 10000 samples
    train_df = train_df.iloc[:TRAIN_SAMPLE_NUM]
    
    # Determine input channels based on embedding type
    input_channels = 3 if embedding_type == 'Naive' else 47  # 3 + 44 for SSPMI
    
    # Preprocess data
    if embedding_type == 'Naive':
        X_train, Y_train = Preprocessing.preprocess_data_naiveemb(
            train_df, ALL_FEATURES, N_KNOWN_FEATURES, train_params,
            shuffle_known_feature=False, shuffle_num=SHUFFLE_NUM
        )
        X_val, Y_val = Preprocessing.preprocess_data_naiveemb(
            val_df, ALL_FEATURES, N_KNOWN_FEATURES, val_params,
            shuffle_known_feature=False, shuffle_num=SHUFFLE_NUM
        )
    else:
        X_train, Y_train = Preprocessing.preprocess_data(
            train_df, ALL_FEATURES, N_KNOWN_FEATURES, train_params,
            SSPMI_EMBEDDING_FILE, shuffle_known_feature=False, shuffle_num=SHUFFLE_NUM
        )
        X_val, Y_val = Preprocessing.preprocess_data(
            val_df, ALL_FEATURES, N_KNOWN_FEATURES, val_params,
            SSPMI_EMBEDDING_FILE, shuffle_known_feature=False, shuffle_num=SHUFFLE_NUM
        )
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model based on type
    if model_type == 'Linear':
        model = LinearModel_Confidence(N_ALL_FEATURES, input_channels, hidden_dim)
    elif model_type == 'CNN':
        model = EncoderDecoderModel_confidence(N_ALL_FEATURES, input_channels, hidden_dim)
    else:  # Transformer
        model = TransformerModel_Confidence(
            input_dim=input_channels,
            embedding_dim=hidden_dim,
            num_heads=4,
            num_layers=3,
            dropout=0.1,
            batch_first=True
        )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Calculate class weights from training data
    all_targets = []
    for _, target in train_loader:
        all_targets.append(target.numpy())
    all_targets = np.concatenate(all_targets)
    
    # Calculate class weights per feature
    num_zeros = np.sum(all_targets == 0, axis=0)  # Sum per feature
    num_ones = np.sum(all_targets == 1, axis=0)
    total = num_zeros + num_ones
    
    # Compute balanced weights per feature
    weights_for_0 = total / (2.0 * num_zeros)
    weights_for_1 = total / (2.0 * num_ones)
    
    # Stack weights into a numpy array first
    class_weights = np.stack([weights_for_0, weights_for_1], axis=0)
    
    # Convert numpy array to tensor
    class_weights = torch.from_numpy(class_weights).to(device=device, dtype=torch.float32)
    
    # Initialize Focal Loss with class weights
    criterion = FocalLoss(alpha=2.0, gamma=5.3)
    
    # Add confidence regularization
    def confidence_aware_loss(predictions, targets, na_mask):
        # Apply both focal loss with class weights
        focal_loss = criterion(predictions, targets, class_weights)
        
        # Add confidence regularization for NA positions
        
        confidence_penalty = torch.abs(predictions[na_mask] - 0.5).mean()
        
        return focal_loss + 0.04 * confidence_penalty   
        # return focal_loss
    
    # Training setup
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=LEARNING_RATE,
        weight_decay=0.005  # Keep L2 regularization
    )
    
    best_val_auc = 0
    train_history = {'train_loss': [], 'train_auc': [], 'val_loss': [], 'val_auc': [], 'iteration_speed': []}
    
    epoch_pbar = tqdm(range(EPOCHS), desc='Training Progress')
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []
        
        batch_pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                         desc=f'Epoch {epoch+1}/{EPOCHS}', leave=False)
        
        for batch_idx, (data, target) in batch_pbar:
            start_time = time.time()
            
            optimizer.zero_grad()
            output = model(data)
            if output.dim() == 3:
                predictions = output[:, :, 0]
            else:
                predictions = output
            
            # Get NA mask for current batch
            batch_na_mask = data[:, :, 0] == 0  # First channel is indicator
            
            # Calculate loss with confidence awareness
            loss = confidence_aware_loss(predictions, target, batch_na_mask)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.append(predictions.detach().cpu().numpy())
            train_targets.append(target.cpu().numpy())
            
            iteration_time = time.time() - start_time
            train_history['iteration_speed'].append(1.0 / iteration_time)
            
            # Update progress bar
            batch_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'speed': f'{1.0/iteration_time:.1f} it/s'
            })
        
        # Calculate training metrics
        train_preds = np.concatenate(train_preds).flatten()
        train_targets = np.concatenate(train_targets).flatten()
        train_auc = roc_auc_score(train_targets, train_preds)
        avg_train_loss = train_loss/len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc='Validation', leave=False)
            for data, target in val_pbar:
                output = model(data)
                if output.dim() == 3:
                    predictions = output[:, :, 0]
                else:
                    predictions = output
                
                loss = confidence_aware_loss(predictions, target, data[:, :, 0] == 0)
                val_loss += loss.item()
                
                val_preds.append(predictions.cpu().numpy())
                val_targets.append(target.cpu().numpy())
        
        # Calculate validation metrics
        val_preds = np.concatenate(val_preds).flatten()
        val_targets = np.concatenate(val_targets).flatten()
        val_auc = roc_auc_score(val_targets, val_preds)
        avg_val_loss = val_loss/len(val_loader)
        
        # Update epoch progress bar with metrics
        epoch_pbar.set_postfix({
            'train_loss': f'{avg_train_loss:.4f}',
            'train_auc': f'{train_auc:.4f}',
            'val_loss': f'{avg_val_loss:.4f}',
            'val_auc': f'{val_auc:.4f}'
        })
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
            }, os.path.join(CHECKPOINT_DIR, f"model_{model_config['model_type']}_{model_config['embedding_type']}_d{model_config['hidden_dim']}_k{N_KNOWN_FEATURES}.pth"))
        
        # Update history
        train_history['train_loss'].append(avg_train_loss)
        train_history['train_auc'].append(train_auc)
        train_history['val_loss'].append(avg_val_loss)
        train_history['val_auc'].append(val_auc)
    
    return train_history

def inference(model_config, temperatures=[0.5, 1.0, 2.0], thresholds=[0.2, 0.3, 0.4, 0.5, 0.6]):
    """Run inference with thresholds targeting training data distribution."""
    # Load model from checkpoint
    checkpoint_name = f"model_{model_config['model_type']}_{model_config['embedding_type']}_d{model_config['hidden_dim']}_k{N_KNOWN_FEATURES}.pth"
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    
    # Model parameters
    input_channels = 3 if model_config['embedding_type'] == 'Naive' else 47
    
    # Initialize model
    if model_config['model_type'] == 'Linear':
        model = LinearModel_Confidence(N_ALL_FEATURES, input_channels, model_config['hidden_dim'])
    elif model_config['model_type'] == 'CNN':
        model = EncoderDecoderModel_confidence(N_ALL_FEATURES, input_channels, model_config['hidden_dim'])
    else:  # Transformer
        model = TransformerModel_Confidence(
            input_dim=input_channels,
            embedding_dim=model_config['hidden_dim'],
            num_heads=4,
            num_layers=3,
            dropout=0.1
        )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test data using inference preprocessing
    test_file = f"../data/2b_Inferenced/Resp_Validation_{N_KNOWN_FEATURES}.csv"
    train_params = Preprocessing.load_item_parameters(TRAIN_PARAMS_FILE, N_ALL_FEATURES)
    
    # Convert parameters to the format expected by inference preprocessing
    item_parameters = {str(i): train_params[f'X{i}'] for i in range(1, N_ALL_FEATURES + 1)}
    all_features = [str(i) for i in range(1, N_ALL_FEATURES + 1)]
    
    # Use the inference preprocessing function
    sspmi_file = SSPMI_EMBEDDING_FILE if model_config['embedding_type'] == 'SSPMI' else None
    input_tensor, original_data = preprocess_inference_data(
        file_path=test_file,
        item_parameters=item_parameters,
        all_features=all_features,
        sspmi_embedding_file=sspmi_file
    )
    
    # Get NA mask from original data
    na_mask = original_data.isna().values
    
    # Calculate target ratios from training data
    train_df = pd.read_csv(TRAIN_FILE, index_col=0)
    train_ratios = []
    for col in train_df.columns:
        total = len(train_df[col].dropna())
        if total > 0:
            ones_ratio = (train_df[col] == 1).sum() / total
            train_ratios.append(ones_ratio)
    
    target_ratio = np.mean(train_ratios)
    print(f"\nTarget ratio from training data (ones): {target_ratio:.4f}")
    
    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        # Handle both 2D and 3D outputs
        if output.dim() == 3:
            raw_preds = output[:, :, 0].cpu().numpy()  # For 3D output (batch, seq, features)
        else:
            raw_preds = output.cpu().numpy()  # For 2D output (batch, seq)
        
        # Print raw prediction statistics
        print("\nRaw prediction statistics:")
        print(f"Mean: {raw_preds.mean():.4f}")
        print(f"Min: {raw_preds.min():.4f}")
        print(f"Max: {raw_preds.max():.4f}")
        
        # Plot distribution
        try:
            plt.figure(figsize=(10, 6))
            
            # Plot both distributions
            plt.hist(raw_preds.flatten(), bins=50, range=(0, 1), density=True, alpha=0.5, 
                    label='All Predictions', color='blue')
            plt.hist(raw_preds[na_mask], bins=50, range=(0, 1), density=True, alpha=0.5,
                    label='NA Predictions', color='orange')
            
            plt.title('Distribution of Raw Predictions')
            plt.xlabel('Predicted Value')
            plt.ylabel('Density')
            plt.grid(True, alpha=0.3)
            
            # Add threshold lines
            for threshold in thresholds:
                plt.axvline(x=threshold, color='r', linestyle='--', alpha=0.5)
            
            # Add mean lines
            plt.axvline(x=raw_preds.mean(), color='blue', linestyle='-',
                       label=f'All Mean = {raw_preds.mean():.3f}')
            plt.axvline(x=raw_preds[na_mask].mean(), color='orange', linestyle='-',
                       label=f'NA Mean = {raw_preds[na_mask].mean():.3f}')
            plt.legend()
            
            plt.savefig(os.path.join(RESULTS_DIR,
                       f'raw_predictions_dist_{model_config["model_type"]}_{model_config["embedding_type"]}_d{model_config["hidden_dim"]}.png'))
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not plot distribution: {str(e)}")
        
        # Analyze temperatures and thresholds
        best_distribution = None
        best_temp = None
        best_threshold = None
        best_balance = float('inf')  # Now measures distance from training distribution
        
        print(f"\nModel: {model_config['model_type']}, Embedding: {model_config['embedding_type']}, Hidden dim: {model_config['hidden_dim']}")
        print("\nDistribution of predictions for NA values:")
        print(f"{'Temperature':>12} {'Threshold':>10} {'Zeros':>10} {'Ones':>10} {'Diff from Target':>15}")
        print("-" * 60)
        
        for temp in temperatures:
            # Apply temperature scaling
            scaled_preds = raw_preds / temp
            
            for threshold in thresholds:
                binary_preds = (scaled_preds > threshold).astype(float)
                binary_preds_reshaped = binary_preds.reshape(na_mask.shape)
                
                # Calculate distribution for NA values only
                na_preds = binary_preds_reshaped[na_mask]
                zeros_ratio = (na_preds == 0).mean()
                ones_ratio = (na_preds == 1).mean()
                
                # Calculate difference from target ratio instead of 0.5
                balance = abs(ones_ratio - target_ratio)
                
                print(f"{temp:12.1f} {threshold:10.2f} {zeros_ratio:10.2%} {ones_ratio:10.2%} {balance:15.4f}")
                
                if balance < best_balance:
                    best_balance = balance
                    best_distribution = (zeros_ratio, ones_ratio)
                    best_temp = temp
                    best_threshold = threshold
        
        print(f"\nBest configuration:")
        print(f"Temperature: {best_temp}")
        print(f"Threshold: {best_threshold}")
        print(f"Distribution - Zeros: {best_distribution[0]:.4f}, Ones: {best_distribution[1]:.4f}")
        print(f"Difference from target ratio: {best_balance:.4f}")
        
        # Save predictions using best temperature and threshold
        best_preds = raw_preds / best_temp
        binary_preds = (best_preds > best_threshold).astype(float)
        binary_preds_reshaped = binary_preds.reshape(na_mask.shape)
        
        result_df = original_data.copy()
        result_df.values[na_mask] = binary_preds_reshaped[na_mask]
        
        output_file = os.path.join(
            INFERENCE_DIR, 
            f"inference_{model_config['model_type']}_{model_config['embedding_type']}_d{model_config['hidden_dim']}.csv"
        )
        result_df.to_csv(output_file)
        
        return {
            'best_temperature': best_temp,
            'best_threshold': best_threshold,
            'zeros_ratio': best_distribution[0],
            'ones_ratio': best_distribution[1],
            'target_ratio': target_ratio,
            'ratio_difference': best_balance,
            'num_predictions': np.sum(na_mask),
            'output_file': os.path.basename(output_file)
        }

def calculate_auc(predictions, targets):
    """Calculate AUC score."""
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()
    return np.mean((predictions > 0.5) == (targets > 0.5))

def save_checkpoint(model, optimizer, config, history):
    """Save model checkpoint and training history."""
    checkpoint_name = f"model_{config['model_type']}_{config['embedding_type']}_d{config['hidden_dim']}_k{N_KNOWN_FEATURES}.pth"
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'history': history
    }, checkpoint_path)

def run_experiments():
    """Run all experiments with different configurations."""
    # Experiment configurations
    hidden_dims = [64, 128, 256]
    model_types = ['Linear', 'CNN', 'Transformer']
    embedding_types = ['Naive', 'SSPMI']
    
    results = []
    results_file = os.path.join(RESULTS_DIR, 'experiment_results.csv')
    
    # Define columns for results
    columns = [
        'model_type',
        'embedding_type',
        'hidden_dim',
        'final_train_auc',
        'final_val_auc',
        'final_train_loss',
        'final_val_loss',
        'avg_iteration_speed',
        'inference_temperature',
        'inference_threshold',
        'inference_zeros_ratio',
        'inference_ones_ratio',
        'inference_target_ratio',
        'inference_ratio_difference',
        'inference_num_predictions',
        'inference_output_file',
        'timestamp'
    ]
    
    # Load existing results if available
    try:
        if os.path.exists(results_file) and os.path.getsize(results_file) > 0:
            existing_results = pd.read_csv(results_file)
            results = existing_results.to_dict('records')
        else:
            # Create empty DataFrame with defined columns
            existing_results = pd.DataFrame(columns=columns)
            results = []
    except pd.errors.EmptyDataError:
        # Handle empty file
        existing_results = pd.DataFrame(columns=columns)
        results = []
    
    completed_experiments = set()
    
    # Create set of completed experiments from existing results
    if not existing_results.empty:
        for _, row in existing_results.iterrows():
            exp_key = f"{row['model_type']}_{row['embedding_type']}_d{row['hidden_dim']}"
            completed_experiments.add(exp_key)
    
    total_experiments = len(hidden_dims) * len(model_types) * len(embedding_types)
    
    # Run experiments with progress bar
    for hidden_dim, model_type, embedding_type in tqdm([(h, m, e) for h in hidden_dims 
                                                       for m in model_types 
                                                       for e in embedding_types], 
                                                      desc="Running Experiments", 
                                                      total=total_experiments):
        # Check if experiment already completed
        exp_key = f"{model_type}_{embedding_type}_d{hidden_dim}"
        if exp_key in completed_experiments:
            print(f"\nSkipping completed experiment: {exp_key}")
            continue
        
        config = {
            'hidden_dim': hidden_dim,
            'model_type': model_type,
            'embedding_type': embedding_type,
            'n_known_features': N_KNOWN_FEATURES
        }
        
        print(f"\nRunning experiment with config: {config}")
        
        try:
            # Train model
            history = train_model(config)
            
            # Run inference
            inference_results = inference(config)
            
            # Store results with precise formatting
            result = {
                'model_type': model_type,
                'embedding_type': embedding_type,
                'hidden_dim': hidden_dim,
                'final_train_auc': f"{history['train_auc'][-1]:.6f}",
                'final_val_auc': f"{history['val_auc'][-1]:.6f}",
                'final_train_loss': f"{history['train_loss'][-1]:.6f}",
                'final_val_loss': f"{history['val_loss'][-1]:.6f}",
                'avg_iteration_speed': f"{np.mean(history['iteration_speed']):.6f}",
                'inference_temperature': f"{inference_results['best_temperature']:.2f}",
                'inference_threshold': f"{inference_results['best_threshold']:.2f}",
                'inference_zeros_ratio': f"{inference_results['zeros_ratio']:.6f}",
                'inference_ones_ratio': f"{inference_results['ones_ratio']:.6f}",
                'inference_target_ratio': f"{inference_results['target_ratio']:.6f}",
                'inference_ratio_difference': f"{inference_results['ratio_difference']:.6f}",
                'inference_num_predictions': int(inference_results['num_predictions']),
                'inference_output_file': inference_results['output_file'],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            results.append(result)
            
            # Save results with clean formatting
            df = pd.DataFrame(results, columns=columns)
            df.to_csv(results_file, index=False, float_format='%.6f')
            
            # Print latest result in a clean format
            print("\nLatest experiment results:")
            for col in columns:
                print(f"{col:25}: {result[col]}")
            
            # Add to completed experiments
            completed_experiments.add(exp_key)
            
            # Analyze training data distribution
            # analyze_training_distribution()
            
        except Exception as e:
            print(f"Error in experiment {exp_key}: {str(e)}")
            # Save error information with minimal fields
            error_result = {
                'model_type': model_type,
                'embedding_type': embedding_type,
                'hidden_dim': hidden_dim,
                'error': str(e),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            results.append(error_result)
            df = pd.DataFrame(results, columns=columns)
            df.to_csv(results_file, index=False)
            continue
    
    return results

def visualize_results(results):
    """Create visualizations for experiment results."""
    df = pd.DataFrame(results)
    
    # 1. Hidden Dimension Comparison
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df, x='hidden_dim', y='final_val_auc', hue='model_type')
    
    # Get x-tick positions
    xticks = ax.get_xticks()
    width = 0.25  # Approximate width of each bar
    
    # Add trend lines for each model type
    for i, model in enumerate(df['model_type'].unique()):
        model_data = df[df['model_type'] == model]
        means = model_data.groupby('hidden_dim')['final_val_auc'].mean()
        # Calculate x positions for this model type's bars
        x_positions = [x + (i - 1) * width for x in xticks]
        plt.plot(x_positions, means.values, '--', alpha=0.6, linewidth=1)
    
    plt.title('Validation AUC vs Hidden Dimension')
    plt.savefig(os.path.join(RESULTS_DIR, 'hidden_dim_comparison.png'))
    plt.close()
    
    # 2. Model Architecture Comparison
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df, x='model_type', y='final_val_auc', hue='embedding_type')
    
    # Get x-tick positions
    xticks = ax.get_xticks()
    width = 0.25  # Approximate width of each bar
    
    # Add trend lines for each embedding type
    for i, embed in enumerate(df['embedding_type'].unique()):
        embed_data = df[df['embedding_type'] == embed]
        means = embed_data.groupby('model_type')['final_val_auc'].mean()
        # Calculate x positions for this embedding type's bars
        x_positions = [x + (i - 0.5) * width for x in xticks]
        plt.plot(x_positions, means.values, '--', alpha=0.6, linewidth=1)
    
    plt.title('Validation AUC vs Model Architecture')
    plt.savefig(os.path.join(RESULTS_DIR, 'model_architecture_comparison.png'))
    plt.close()
    
    # 3. Embedding Type Comparison
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df, x='embedding_type', y='final_val_auc', hue='model_type')
    
    # Get x-tick positions
    xticks = ax.get_xticks()
    width = 0.25  # Approximate width of each bar
    
    # Add trend lines for each model type
    for i, model in enumerate(df['model_type'].unique()):
        model_data = df[df['model_type'] == model]
        means = model_data.groupby('embedding_type')['final_val_auc'].mean()
        # Calculate x positions for this model type's bars
        x_positions = [x + (i - 1) * width for x in xticks]
        plt.plot(x_positions, means.values, '--', alpha=0.6, linewidth=1)
    
    plt.title('Validation AUC vs Embedding Type')
    plt.savefig(os.path.join(RESULTS_DIR, 'embedding_type_comparison.png'))
    plt.close()
    
    # 4. Speed Comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='model_type', y='avg_iteration_speed')
    plt.title('Average Iteration Speed by Model Type')
    plt.savefig(os.path.join(RESULTS_DIR, 'speed_comparison.png'))
    plt.close()
    
    # 5. Inference Analysis
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='inference_ones_ratio', y='inference_zeros_ratio', 
                    hue='model_type', style='embedding_type')
    plt.title('Inference Ones Ratio vs Zeros Ratio')
    plt.savefig(os.path.join(RESULTS_DIR, 'inference_ratio_analysis.png'))
    plt.close()
    
    # Save numerical results
    df.to_csv(os.path.join(RESULTS_DIR, 'experiment_results.csv'), index=False)

def analyze_training_distribution():
    """Analyze the distribution of values in training data."""
    print("\nAnalyzing training data distribution...")
    
    # Load training data
    train_df = pd.read_csv(TRAIN_FILE, index_col=0)
    
    # Calculate overall statistics
    total_values = train_df.size
    total_ones = (train_df == 1).sum().sum()
    total_zeros = (train_df == 0).sum().sum()
    
    print("\nOverall Training Data Distribution:")
    print(f"Total values: {total_values}")
    print(f"Zeros: {total_zeros} ({total_zeros/total_values:.2%})")
    print(f"Ones: {total_ones} ({total_ones/total_values:.2%})")
    
    # Analyze distribution per feature
    print("\nDistribution per feature:")
    print(f"{'Feature':>10} {'Zeros %':>10} {'Ones %':>10} {'Total':>10}")
    print("-" * 45)
    
    for col in train_df.columns:
        total = len(train_df[col].dropna())
        if total > 0:  # Only analyze if feature has values
            zeros = (train_df[col] == 0).sum()
            ones = (train_df[col] == 1).sum()
            print(f"{col:>10} {zeros/total:>9.2%} {ones/total:>9.2%} {total:>10}")

def preprocess_inference_data(file_path, item_parameters, all_features, sspmi_embedding_file=None):
    """
    Preprocess the inference dataset for model input.
    This is the same function as in inference.py but renamed to avoid confusion.
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
        # For Naive embedding, reshape to match dimensions
        embeddings = np.array([item_parameters[col] for col in all_features])
        embeddings = np.tile(embeddings, (num_samples, 1))  # Shape: (num_samples, num_all_features)
        embeddings = embeddings[:, :, np.newaxis]  # Add extra dimension to match shape

    # Combine indicator, values, and embeddings into a single input tensor
    input_tensor = np.concatenate([
        indicator[:, :, np.newaxis],  # Add dimension to match
        values[:, :, np.newaxis],     # Add dimension to match
        embeddings
    ], axis=-1)

    return torch.tensor(input_tensor, dtype=torch.float32), data

if __name__ == "__main__":
    print("Starting experiments...")
    results = run_experiments()
    print("\nGenerating visualizations...")
    visualize_results(results)
    print("\nExperiments completed. Results saved in the 'results' directory.")
