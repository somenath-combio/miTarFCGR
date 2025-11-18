# miTarFCGR - miRNA-mRNA Interaction Prediction

This project uses Frequency Chaos Game Representation (FCGR) and deep learning to predict miRNA-mRNA interactions.

## Project Structure

```
miTarFCGR/
├── data/
│   └── miraw.csv                # Raw miRNA-mRNA interaction data
├── core/
│   └── main.py                  # Model architecture (InteractionModel)
├── utils/
│   └── fcgr.py                  # FCGR calculation utilities
├── train.py                     # Main training script with GPU support
├── check_gpu.py                 # GPU diagnostic script
├── verify_setup.py              # Setup verification script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── GPU_TROUBLESHOOTING.md       # GPU troubleshooting guide
```

## Installation

### 1. Install PyTorch with CUDA Support (for GPU training)

For **NVIDIA RTX 5080** or other NVIDIA GPUs:

```bash
# For CUDA 12.1 (Recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Or for CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Install Other Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify GPU Setup

```bash
python check_gpu.py
```

This will check if your GPU is properly detected and configured. Expected output:
```
✓ CUDA is available!
  Name: NVIDIA GeForce RTX 5080
  Total Memory: 16.00 GB
```

**Important:** If GPU is not detected, see [GPU_TROUBLESHOOTING.md](GPU_TROUBLESHOOTING.md) for detailed troubleshooting steps.

## Usage

### Training the Model

To train the model with default parameters:

```bash
python train.py
```

### Configuration Parameters

The training script uses the following default parameters:

- **K-mer size (K)**: 6 (generates 64×64 FCGR)
- **Batch size**: 64
- **Learning rate**: 0.003
- **Dropout rate**: 0.1
- **Number of epochs**: 100
- **Loss function**: CrossEntropyLoss
- **Optimizer**: Adam
- **Mixed Precision**: Enabled (for faster GPU training)
- **Device**: Auto-detected (CUDA if available, otherwise CPU)

### Modifying Parameters

You can modify the parameters in the `train.py` file under the `if __name__ == "__main__":` section:

```python
# Configuration
CSV_PATH = "data/miraw.csv"
K = 6                      # K-mer size for FCGR
BATCH_SIZE = 64            # Batch size for training
LEARNING_RATE = 0.003      # Learning rate
DROPOUT_RATE = 0.1         # Dropout rate
NUM_EPOCHS = 100            # Number of training epochs
MODEL_SAVE_PATH = "best_model.pth"  # Path to save the best model
```

## Data Format

The input CSV file should contain the following columns:

- `mature_miRNA_Transcript`: miRNA sequence (RNA format with U)
- `mRNA_Site_Transcript`: mRNA sequence (RNA format with U)
- `validation`: Binary label (0 or 1) indicating interaction
- `is_valid_mRNA` (optional): Boolean indicating valid mRNA sequence
- `is_valid_miRNA` (optional): Boolean indicating valid miRNA sequence

## Training Process

The training script performs the following steps:

1. **Data Loading**: Reads the CSV file using pandas
2. **Data Preprocessing**:
   - Removes duplicates based on miRNA, mRNA, and validation columns
   - Filters valid sequences
   - Splits data into training (80%) and validation (20%) sets
3. **Sequence Processing**:
   - Converts RNA sequences to DNA (U → T)
   - Calculates FCGR representations for both miRNA and mRNA
4. **Model Training**:
   - Uses the InteractionModel architecture
   - Monitors both training and validation metrics
   - Saves the best model based on validation accuracy

## Model Architecture

The model uses:
- **ModelK6**: CNN architecture for 64×64 FCGR input (k=6)
- **Inception blocks**: Multi-scale feature extraction
- **Dual-branch architecture**: Separate processing for miRNA and mRNA
- **Fully connected layers**: Feature fusion and classification

## Output

During training, the script will:
- Display training progress for each epoch
- Show training and validation loss/accuracy
- Save the best model to `best_model.pth`
- Log all important information

Example output:
```
Epoch [1/100]
----------------------------------------
Batch [10/100], Loss: 0.6234
...
Training   - Loss: 0.5834, Accuracy: 68.45%
Validation - Loss: 0.5421, Accuracy: 72.31%
✓ New best model saved with validation accuracy: 72.31%
```

## Model Checkpoints

The best model is saved with the following information:
- Model state dictionary
- Optimizer state dictionary
- Validation accuracy and loss
- K-mer size and dropout rate
- Epoch number

## Loading a Trained Model

To load a saved model:

```python
import torch
from core.main import InteractionModel

# Load checkpoint
checkpoint = torch.load('best_model.pth')

# Initialize model
model = InteractionModel(
    dropout_rate=checkpoint['dropout_rate'],
    k=checkpoint['k']
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Loaded model with validation accuracy: {checkpoint['val_accuracy']:.2f}%")
```

## GPU Training Features

The training script includes **full GPU support** with the following optimizations:

### Automatic GPU Detection
- Automatically detects and uses NVIDIA GPU if available
- Falls back to CPU if GPU is not detected
- Displays detailed GPU information at startup

### Mixed Precision Training (AMP)
- Uses PyTorch's Automatic Mixed Precision (AMP) for faster training
- Reduces memory usage and increases throughput
- Automatically enabled for GPU training

### GPU Memory Management
- Monitors and logs GPU memory usage after each epoch
- Tracks allocated, reserved, and peak memory
- Helps identify memory issues early

### Performance Optimizations
- **Pin Memory**: Enabled for faster CPU-to-GPU data transfer
- **Non-blocking Transfers**: Asynchronous data movement to GPU
- **cuDNN Benchmarking**: Auto-tuning for optimal performance
- **Persistent Workers**: DataLoader workers stay alive between epochs
- **Prefetching**: Loads next batch while training current batch

### Example GPU Output

```
================================================================================
GPU Configuration
================================================================================
✓ CUDA is available!
  PyTorch version: 2.1.0
  CUDA version: 12.1
  cuDNN version: 8902
  Number of GPUs: 1

  GPU 0:
    Name: NVIDIA GeForce RTX 5080
    Compute Capability: 8.9
    Total Memory: 16.00 GB
    Multi-Processors: 82

✓ Using device: cuda:0 (NVIDIA GeForce RTX 5080)
✓ cuDNN benchmarking enabled for optimal performance
✓ GPU cache cleared

[GPU Memory - After model loading]
  Allocated: 0.45 GB | Reserved: 0.48 GB | Peak: 0.45 GB

Epoch training time: 42.15s (13.2 batches/sec)

[GPU Memory - After Epoch 1]
  Allocated: 2.31 GB | Reserved: 2.50 GB | Peak: 3.04 GB
```

### Troubleshooting GPU Issues

If your GPU is not being used:

1. **Check GPU detection:**
   ```bash
   python check_gpu.py
   ```

2. **Verify CUDA installation:**
   ```bash
   nvidia-smi
   ```

3. **For detailed troubleshooting:**
   See [GPU_TROUBLESHOOTING.md](GPU_TROUBLESHOOTING.md) for comprehensive solutions

### Performance Tips for RTX 5080

- **Recommended batch size:** 64-128 (adjust based on available memory)
- **Expected speed:** ~10-20 batches/second
- **GPU utilization:** Should be 80-95%
- **Memory usage:** ~2-4 GB for k=6, batch_size=64

## Validation Accuracy Monitoring

The script continuously monitors validation accuracy during training:
- Evaluates on the validation set after each epoch
- Saves the model only when validation accuracy improves
- Prevents overfitting by tracking generalization performance

## Notes

- The script automatically detects and uses GPU if available
- Mixed precision training is enabled by default for GPU
- Data is automatically shuffled during training
- Sequences containing 'N' nucleotides are handled with warnings
- Invalid k-mers are skipped during FCGR calculation
- GPU memory is monitored and logged throughout training
