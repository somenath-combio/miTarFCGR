# miTarFCGR - miRNA-mRNA Interaction Prediction

This project uses Frequency Chaos Game Representation (FCGR) and deep learning to predict miRNA-mRNA interactions.

## Project Structure

```
miTarFCGR/
├── data/
│   └── miraw.csv           # Raw miRNA-mRNA interaction data
├── core/
│   └── main.py             # Model architecture (InteractionModel)
├── utils/
│   └── fcgr.py             # FCGR calculation utilities
├── train.py                # Main training script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

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
- **Number of epochs**: 50
- **Loss function**: CrossEntropyLoss
- **Optimizer**: Adam

### Modifying Parameters

You can modify the parameters in the `train.py` file under the `if __name__ == "__main__":` section:

```python
# Configuration
CSV_PATH = "data/miraw.csv"
K = 6                      # K-mer size for FCGR
BATCH_SIZE = 64            # Batch size for training
LEARNING_RATE = 0.003      # Learning rate
DROPOUT_RATE = 0.1         # Dropout rate
NUM_EPOCHS = 50            # Number of training epochs
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
Epoch [1/50]
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

## Validation Accuracy Monitoring

The script continuously monitors validation accuracy during training:
- Evaluates on the validation set after each epoch
- Saves the model only when validation accuracy improves
- Prevents overfitting by tracking generalization performance

## Notes

- The script automatically detects and uses GPU if available
- Data is automatically shuffled during training
- Sequences containing 'N' nucleotides are handled with warnings
- Invalid k-mers are skipped during FCGR calculation
