# GPU Troubleshooting Guide for RTX 5080

This guide helps you troubleshoot GPU issues when training the miRNA-mRNA interaction model.

## Quick Diagnostics

### Step 1: Run the GPU Check Script

```bash
python check_gpu.py
```

This will verify:
- ✓ PyTorch installation with CUDA support
- ✓ GPU detection and availability
- ✓ CUDA and cuDNN versions
- ✓ GPU memory and specifications
- ✓ Basic GPU computation test

### Expected Output for RTX 5080

```
============================================================
GPU Configuration
============================================================
✓ CUDA is available!
  PyTorch version: 2.x.x
  CUDA version: 12.x
  cuDNN version: 8xxx
  Number of GPUs: 1

  GPU 0:
    Name: NVIDIA GeForce RTX 5080
    Compute Capability: 8.9
    Total Memory: 16.00 GB (or your GPU's memory)
    Multi-Processors: XX
```

## Common Issues and Solutions

### Issue 1: "CUDA is not available"

**Symptoms:**
- Training uses CPU instead of GPU
- `torch.cuda.is_available()` returns `False`

**Solutions:**

#### A. Install PyTorch with CUDA Support

```bash
# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio

# Install with CUDA 12.1 (recommended for RTX 5080)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Or CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### B. Verify NVIDIA Drivers

```bash
# Check NVIDIA driver version
nvidia-smi

# Expected output should show your RTX 5080
```

If `nvidia-smi` fails:
1. Install/update NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx
2. Reboot your system
3. Try again

#### C. Check CUDA Toolkit

```bash
# Check CUDA version
nvcc --version

# Or
nvidia-smi
```

If CUDA is not installed:
1. Download from: https://developer.nvidia.com/cuda-downloads
2. Install CUDA Toolkit 12.1 or 11.8
3. Reboot

### Issue 2: GPU Detected but Not Used During Training

**Symptoms:**
- `check_gpu.py` passes
- Training still uses CPU
- GPU utilization is 0%

**Solutions:**

#### A. Verify Model is on GPU

Add this debug code after model initialization in train.py:

```python
# After line: model = model.to(device)
print(f"Model device: {next(model.parameters()).device}")
print(f"Expected device: {device}")
```

Should output:
```
Model device: cuda:0
Expected device: cuda:0
```

#### B. Check Data Transfer

Add debug code in training loop:

```python
# In train_epoch function
print(f"Input device: {mirna_fcgr.device}")
print(f"Target device: {labels.device}")
```

Should output:
```
Input device: cuda:0
Target device: cuda:0
```

### Issue 3: Out of Memory (OOM) Errors

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

#### A. Reduce Batch Size

In `train.py`, change:
```python
BATCH_SIZE = 64  # Try 32, 16, or 8
```

#### B. Enable Gradient Accumulation

```python
# In train.py configuration
BATCH_SIZE = 32
ACCUMULATION_STEPS = 2  # Effective batch size = 64
```

Then modify training loop to accumulate gradients.

#### C. Clear GPU Cache

```bash
# Before training
python -c "import torch; torch.cuda.empty_cache()"
```

#### D. Monitor GPU Memory

```bash
# In another terminal during training
watch -n 1 nvidia-smi
```

### Issue 4: Slow GPU Training

**Symptoms:**
- Training is slower than expected
- GPU utilization < 80%

**Solutions:**

#### A. Enable Mixed Precision (Already Enabled)

Verify in train.py:
```python
USE_AMP = True  # Should be True
```

#### B. Increase Num Workers

In train.py, modify `prepare_dataloaders`:
```python
num_workers = 8  # Try 4, 8, or 16 (was 4)
```

#### C. Increase Batch Size

```python
BATCH_SIZE = 128  # If you have enough GPU memory
```

#### D. Enable cuDNN Autotuner (Already Enabled)

Verify in code:
```python
torch.backends.cudnn.benchmark = True  # Should be True
```

### Issue 5: CUDA Version Mismatch

**Symptoms:**
```
RuntimeError: CUDA version mismatch
AssertionError: Torch not compiled with CUDA enabled
```

**Solutions:**

#### A. Match PyTorch CUDA Version with System CUDA

```bash
# Check system CUDA
nvidia-smi  # Look at CUDA Version in top right

# Install matching PyTorch
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Monitoring GPU During Training

### Method 1: nvidia-smi

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Or with grep for cleaner output
watch -n 1 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv'
```

### Method 2: nvtop (More Visual)

```bash
# Install nvtop
sudo apt install nvtop  # Ubuntu/Debian
sudo dnf install nvtop  # Fedora

# Run
nvtop
```

### Method 3: Python Script

```python
import torch
import time

while True:
    print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
    time.sleep(1)
```

## Performance Optimization Tips

### 1. Optimal Batch Size for RTX 5080

```python
# Start with these and adjust based on memory:
K=6: BATCH_SIZE = 64-128
K=5: BATCH_SIZE = 128-256
K=4: BATCH_SIZE = 256-512
```

### 2. Enable TensorFloat32 (TF32)

Add to train.py at the beginning:
```python
# For RTX 5080, enable TF32 for better performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### 3. Pin Memory and Async Transfer

Already enabled in the code:
```python
pin_memory=True
non_blocking=True  # In .to(device) calls
```

### 4. Persistent Workers

Already enabled:
```python
persistent_workers=True
```

## Verifying GPU is Working

During training, you should see:

1. **Console Output:**
```
GPU Configuration
================================================================================
✓ CUDA is available!
  Name: NVIDIA GeForce RTX 5080
  Total Memory: 16.00 GB
✓ Using device: cuda:0
```

2. **GPU Memory Usage:**
```
[GPU Memory - After model loading]
  Allocated: 0.50 GB | Reserved: 0.52 GB | Peak: 0.50 GB

[GPU Memory - After Epoch 1]
  Allocated: 2.34 GB | Reserved: 2.50 GB | Peak: 3.12 GB
```

3. **Training Speed:**
```
Epoch training time: 45.23s (12.3 batches/sec)  # Should be fast on GPU
```

4. **nvidia-smi Shows:**
- GPU Utilization: 80-100%
- Memory Usage: Several GB allocated
- Process: python

## Getting Help

If issues persist:

1. **Gather Information:**
```bash
# Save diagnostic info
python check_gpu.py > gpu_diagnostic.txt
nvidia-smi > nvidia_info.txt
nvcc --version > cuda_info.txt
pip list | grep torch > pytorch_info.txt
```

2. **Check PyTorch Forums:**
- https://discuss.pytorch.org/

3. **NVIDIA Developer Forums:**
- https://forums.developer.nvidia.com/

## Quick Reference Commands

```bash
# Check GPU
nvidia-smi

# Monitor GPU in real-time
watch -n 1 nvidia-smi

# Check CUDA
nvcc --version

# Check PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Run diagnostic
python check_gpu.py

# Test training (short)
python train.py  # Press Ctrl+C after 1 epoch to test
```

## Expected Training Performance

For RTX 5080 with the default configuration:
- **Batch size:** 64
- **K-mer:** 6 (64×64 FCGR)
- **Mixed Precision:** Enabled
- **Expected speed:** ~10-20 batches/second
- **Memory usage:** ~2-4 GB
- **GPU utilization:** 80-95%

If your performance is significantly different, revisit the troubleshooting steps above.
