import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.main import InteractionModel
from utils.fcgr import FCGR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class miRNADataset(Dataset):
    """
    PyTorch Dataset for miRNA-mRNA interaction prediction
    """
    def __init__(self, mirna_sequences, mrna_sequences, labels, k=6):
        """
        Args:
            mirna_sequences: List of miRNA sequences
            mrna_sequences: List of mRNA sequences
            labels: List of validation labels (0 or 1)
            k: k-mer size for FCGR calculation
        """
        self.mirna_sequences = mirna_sequences
        self.mrna_sequences = mrna_sequences
        self.labels = labels
        self.k = k

        logging.info(f"Dataset initialized with {len(self.labels)} samples, k={k}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns:
            mirna_fcgr: FCGR representation of miRNA (1, 64, 64) for k=6
            mrna_fcgr: FCGR representation of mRNA (1, 64, 64) for k=6
            label: Binary label (0 or 1)
        """
        # Convert RNA to DNA (U -> T)
        mirna_seq = self.mirna_sequences[idx].replace('U', 'T')
        mrna_seq = self.mrna_sequences[idx].replace('U', 'T')

        # Calculate FCGR
        try:
            mirna_fcgr_obj = FCGR(sequence=mirna_seq, k=self.k)
            mrna_fcgr_obj = FCGR(sequence=mrna_seq, k=self.k)

            mirna_fcgr = mirna_fcgr_obj.generate_fcgr()
            mrna_fcgr = mrna_fcgr_obj.generate_fcgr()

            # Add channel dimension (B, C, H, W) - we add channel later
            mirna_fcgr = torch.FloatTensor(mirna_fcgr).unsqueeze(0)  # (1, 64, 64)
            mrna_fcgr = torch.FloatTensor(mrna_fcgr).unsqueeze(0)    # (1, 64, 64)

            label = torch.LongTensor([self.labels[idx]])[0]

            return mirna_fcgr, mrna_fcgr, label

        except Exception as e:
            logging.error(f"Error processing sample {idx}: {e}")
            logging.error(f"miRNA sequence: {mirna_seq[:50]}...")
            logging.error(f"mRNA sequence: {mrna_seq[:50]}...")
            raise


def setup_gpu():
    """
    Setup and verify GPU configuration
    Returns device to use for training
    """
    logging.info("=" * 80)
    logging.info("GPU Configuration")
    logging.info("=" * 80)

    # Check CUDA availability
    if not torch.cuda.is_available():
        logging.warning("CUDA is not available! Training will use CPU.")
        logging.warning("This will be significantly slower.")
        logging.warning("To use GPU, install PyTorch with CUDA support:")
        logging.warning("Visit https://pytorch.org/get-started/locally/")
        return torch.device('cpu')

    # CUDA is available
    logging.info(f"✓ CUDA is available!")
    logging.info(f"  PyTorch version: {torch.__version__}")
    logging.info(f"  CUDA version: {torch.version.cuda}")
    logging.info(f"  cuDNN version: {torch.backends.cudnn.version()}")
    logging.info(f"  Number of GPUs: {torch.cuda.device_count()}")

    # Display GPU information
    for i in range(torch.cuda.device_count()):
        logging.info(f"\n  GPU {i}:")
        logging.info(f"    Name: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        logging.info(f"    Compute Capability: {props.major}.{props.minor}")
        logging.info(f"    Total Memory: {props.total_memory / 1024**3:.2f} GB")
        logging.info(f"    Multi-Processors: {props.multi_processor_count}")

    # Set device
    device = torch.device('cuda:0')
    logging.info(f"\n✓ Using device: {device} ({torch.cuda.get_device_name(0)})")

    # Enable cuDNN benchmarking for better performance
    torch.backends.cudnn.benchmark = True
    logging.info("✓ cuDNN benchmarking enabled for optimal performance")

    # Clear GPU cache
    torch.cuda.empty_cache()
    logging.info("✓ GPU cache cleared")

    logging.info("=" * 80)
    return device


def log_gpu_memory(device, stage=""):
    """
    Log current GPU memory usage
    """
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3

        if stage:
            logging.info(f"[GPU Memory - {stage}]")
        logging.info(f"  Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | Peak: {max_allocated:.2f} GB")


def load_and_preprocess_data(csv_path: str) -> pd.DataFrame:
    """
    Load and preprocess the miRNA-mRNA interaction data

    Args:
        csv_path: Path to the CSV file

    Returns:
        Preprocessed DataFrame
    """
    logging.info(f"Loading data from {csv_path}")

    # Load CSV
    df = pd.read_csv(csv_path)
    logging.info(f"Loaded {len(df)} rows from CSV")

    # Check required columns
    required_columns = ['mature_miRNA_Transcript', 'mRNA_Site_Transcript', 'validation']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV")

    # Drop duplicates based on miRNA, mRNA, and validation
    initial_count = len(df)
    df = df.drop_duplicates(subset=['mature_miRNA_Transcript', 'mRNA_Site_Transcript', 'validation'])
    logging.info(f"Dropped {initial_count - len(df)} duplicate rows, {len(df)} rows remaining")

    # Filter valid sequences
    if 'is_valid_mRNA' in df.columns and 'is_valid_miRNA' in df.columns:
        df = df[df['is_valid_mRNA'] & df['is_valid_miRNA']]
        logging.info(f"Filtered to {len(df)} rows with valid sequences")

    # Remove rows with missing values
    df = df.dropna(subset=required_columns)
    logging.info(f"After removing NaN values: {len(df)} rows")

    return df


def prepare_dataloaders(df: pd.DataFrame, k: int = 6, batch_size: int = 64,
                       test_size: float = 0.2, random_state: int = 42,
                       use_gpu: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare training and validation dataloaders

    Args:
        df: Preprocessed DataFrame
        k: k-mer size for FCGR
        batch_size: Batch size for DataLoader
        test_size: Proportion of data to use for validation
        random_state: Random seed for reproducibility
        use_gpu: Whether GPU is being used

    Returns:
        train_loader, val_loader
    """
    logging.info("Preparing dataloaders...")

    # Extract sequences and labels
    mirna_sequences = df['mature_miRNA_Transcript'].values
    mrna_sequences = df['mRNA_Site_Transcript'].values
    labels = df['validation'].values

    # Split into train and validation sets
    mirna_train, mirna_val, mrna_train, mrna_val, y_train, y_val = train_test_split(
        mirna_sequences, mrna_sequences, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    logging.info(f"Train set: {len(y_train)} samples")
    logging.info(f"Validation set: {len(y_val)} samples")
    logging.info(f"Train label distribution: {np.bincount(y_train)}")
    logging.info(f"Validation label distribution: {np.bincount(y_val)}")

    # Create datasets
    train_dataset = miRNADataset(mirna_train, mrna_train, y_train, k=k)
    val_dataset = miRNADataset(mirna_val, mrna_val, y_val, k=k)

    # Optimize num_workers for GPU
    num_workers = 4 if use_gpu else 0

    # Create dataloaders with GPU optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_gpu,  # Pin memory for faster GPU transfer
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_gpu,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    return train_loader, val_loader


def train_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module,
                optimizer: torch.optim.Optimizer, device: torch.device,
                scaler: GradScaler = None, use_amp: bool = False) -> Tuple[float, float]:
    """
    Train the model for one epoch with optional mixed precision

    Returns:
        average_loss, accuracy
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    for batch_idx, (mirna_fcgr, mrna_fcgr, labels) in enumerate(train_loader):
        # Move to device (non_blocking for async GPU transfer)
        mirna_fcgr = mirna_fcgr.to(device, non_blocking=True)
        mrna_fcgr = mrna_fcgr.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

        # Mixed precision training
        if use_amp and scaler is not None:
            with autocast():
                outputs = model(mrna_fcgr, mirna_fcgr)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            outputs = model(mrna_fcgr, mirna_fcgr)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (batch_idx + 1) % 10 == 0:
            current_acc = 100.0 * correct / total
            logging.info(f"Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, Acc: {current_acc:.2f}%")

    elapsed_time = time.time() - start_time
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    logging.info(f"Epoch training time: {elapsed_time:.2f}s ({len(train_loader)/elapsed_time:.2f} batches/sec)")

    return avg_loss, accuracy


def validate(model: nn.Module, val_loader: DataLoader, criterion: nn.Module,
            device: torch.device, use_amp: bool = False) -> Tuple[float, float]:
    """
    Validate the model with optional mixed precision

    Returns:
        average_loss, accuracy
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for mirna_fcgr, mrna_fcgr, labels in val_loader:
            # Move to device (non_blocking for async GPU transfer)
            mirna_fcgr = mirna_fcgr.to(device, non_blocking=True)
            mrna_fcgr = mrna_fcgr.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Mixed precision inference
            if use_amp:
                with autocast():
                    outputs = model(mrna_fcgr, mirna_fcgr)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(mrna_fcgr, mirna_fcgr)
                loss = criterion(outputs, labels)

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def train_model(csv_path: str, k: int = 6, batch_size: int = 64, learning_rate: float = 0.003,
                dropout_rate: float = 0.1, num_epochs: int = 100, model_save_path: str = "best_model.pth",
                use_amp: bool = True):
    """
    Main training function with GPU support

    Args:
        csv_path: Path to the CSV data file
        k: k-mer size for FCGR
        batch_size: Batch size
        learning_rate: Learning rate
        dropout_rate: Dropout rate
        num_epochs: Number of training epochs
        model_save_path: Path to save the best model
        use_amp: Use Automatic Mixed Precision for faster GPU training
    """
    logging.info("=" * 80)
    logging.info("Starting Training - miRNA-mRNA Interaction Prediction")
    logging.info("=" * 80)
    logging.info(f"Parameters:")
    logging.info(f"  K-mer size: {k}")
    logging.info(f"  Batch size: {batch_size}")
    logging.info(f"  Learning rate: {learning_rate}")
    logging.info(f"  Dropout rate: {dropout_rate}")
    logging.info(f"  Number of epochs: {num_epochs}")
    logging.info(f"  Mixed Precision (AMP): {use_amp}")
    logging.info("=" * 80)

    # Setup GPU
    device = setup_gpu()
    use_gpu = device.type == 'cuda'

    # Disable AMP if not using GPU
    if not use_gpu:
        use_amp = False
        logging.info("Mixed precision disabled (CPU mode)")

    # Load and preprocess data
    df = load_and_preprocess_data(csv_path)

    # Prepare dataloaders
    train_loader, val_loader = prepare_dataloaders(df, k=k, batch_size=batch_size, use_gpu=use_gpu)

    # Initialize model
    logging.info("\nInitializing model...")
    model = InteractionModel(dropout_rate=dropout_rate, k=k)

    # Move model to GPU
    model = model.to(device)
    logging.info(f"✓ Model moved to {device}")

    # Log GPU memory after model loading
    if use_gpu:
        log_gpu_memory(device, "After model loading")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")

    # Estimate model size
    param_size = total_params * 4 / 1024**2  # Assuming float32 (4 bytes)
    logging.info(f"Estimated model size: {param_size:.2f} MB")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize gradient scaler for mixed precision
    scaler = GradScaler() if use_amp else None
    if use_amp:
        logging.info("✓ Gradient scaler initialized for mixed precision training")

    # Training loop
    best_val_accuracy = 0.0
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    logging.info("=" * 80)
    logging.info("Starting Training Loop")
    logging.info("=" * 80)

    for epoch in range(num_epochs):
        logging.info(f"\n{'='*80}")
        logging.info(f"Epoch [{epoch + 1}/{num_epochs}]")
        logging.info(f"{'='*80}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp)
        logging.info(f"Training   - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, use_amp)
        logging.info(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

        # Log GPU memory after epoch
        if use_gpu:
            log_gpu_memory(device, f"After Epoch {epoch + 1}")

        # Save history
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)

        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'val_loss': val_loss,
                'train_accuracy': train_acc,
                'train_loss': train_loss,
                'k': k,
                'dropout_rate': dropout_rate,
                'batch_size': batch_size,
                'learning_rate': learning_rate
            }, model_save_path)
            logging.info(f"✓ New best model saved with validation accuracy: {val_acc:.2f}%")

    logging.info("\n" + "=" * 80)
    logging.info("Training Complete!")
    logging.info("=" * 80)
    logging.info(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")
    logging.info(f"Model saved to: {model_save_path}")

    if use_gpu:
        logging.info("\nFinal GPU Memory Usage:")
        log_gpu_memory(device, "Training Complete")

    logging.info("=" * 80)

    return model, training_history


if __name__ == "__main__":
    # Configuration
    CSV_PATH = "data/miraw.csv"
    K = 6
    BATCH_SIZE = 64
    LEARNING_RATE = 0.003
    DROPOUT_RATE = 0.1
    NUM_EPOCHS = 100
    MODEL_SAVE_PATH = "best_model.pth"
    USE_AMP = True  # Use Automatic Mixed Precision for faster GPU training

    # Train the model
    model, history = train_model(
        csv_path=CSV_PATH,
        k=K,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        dropout_rate=DROPOUT_RATE,
        num_epochs=NUM_EPOCHS,
        model_save_path=MODEL_SAVE_PATH,
        use_amp=USE_AMP
    )
