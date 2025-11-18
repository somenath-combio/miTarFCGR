#!/usr/bin/env python3
"""
GPU Diagnostic Script
Run this script to check if PyTorch can detect and use your GPU
"""

import sys

def check_gpu():
    print("=" * 80)
    print("GPU Diagnostic Check")
    print("=" * 80)

    # Check PyTorch installation
    try:
        import torch
        print(f"✓ PyTorch installed: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not installed!")
        print("  Install with: pip install torch torchvision torchaudio")
        return False

    # Check CUDA availability
    print(f"\nCUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\n✗ CUDA is not available!")
        print("\nPossible reasons:")
        print("  1. PyTorch was installed without CUDA support")
        print("  2. NVIDIA drivers are not installed")
        print("  3. CUDA toolkit is not installed")
        print("\nTo install PyTorch with CUDA support:")
        print("  Visit: https://pytorch.org/get-started/locally/")
        print("  Example: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return False

    # Display CUDA information
    print(f"\n✓ CUDA is available!")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"  Number of GPUs: {torch.cuda.device_count()}")

    # Display GPU details
    for i in range(torch.cuda.device_count()):
        print(f"\n--- GPU {i} ---")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")

        # Memory information
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  Total Memory: {total_memory:.2f} GB")

        # Current memory usage
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  Allocated Memory: {allocated:.2f} GB")
        print(f"  Reserved Memory: {reserved:.2f} GB")

    # Test GPU computation
    print("\n" + "=" * 80)
    print("Testing GPU Computation")
    print("=" * 80)

    try:
        # Create tensors
        print("\n1. Creating test tensors...")
        x = torch.randn(1000, 1000)
        y = torch.randn(1000, 1000)
        print("  ✓ Tensors created on CPU")

        # Move to GPU
        print("\n2. Moving tensors to GPU...")
        device = torch.device('cuda:0')
        x_gpu = x.to(device)
        y_gpu = y.to(device)
        print(f"  ✓ Tensors moved to {device}")

        # Perform computation
        print("\n3. Performing matrix multiplication on GPU...")
        import time
        start = time.time()
        z_gpu = torch.matmul(x_gpu, y_gpu)
        torch.cuda.synchronize()  # Wait for GPU to finish
        gpu_time = time.time() - start
        print(f"  ✓ GPU computation completed in {gpu_time*1000:.2f} ms")

        # Compare with CPU
        print("\n4. Comparing with CPU computation...")
        start = time.time()
        z_cpu = torch.matmul(x, y)
        cpu_time = time.time() - start
        print(f"  ✓ CPU computation completed in {cpu_time*1000:.2f} ms")
        print(f"  → GPU is {cpu_time/gpu_time:.2f}x faster")

        # Test neural network
        print("\n5. Testing neural network on GPU...")
        model = torch.nn.Sequential(
            torch.nn.Linear(1000, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 100)
        ).to(device)

        input_tensor = torch.randn(64, 1000).to(device)
        output = model(input_tensor)
        print(f"  ✓ Neural network forward pass successful")
        print(f"  → Input shape: {input_tensor.shape}")
        print(f"  → Output shape: {output.shape}")

        print("\n" + "=" * 80)
        print("✓ All GPU tests passed successfully!")
        print("=" * 80)
        print("\nYour GPU is ready for training!")
        print(f"Recommended device: cuda:0 ({torch.cuda.get_device_name(0)})")

        return True

    except Exception as e:
        print(f"\n✗ GPU test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = check_gpu()
    sys.exit(0 if success else 1)
