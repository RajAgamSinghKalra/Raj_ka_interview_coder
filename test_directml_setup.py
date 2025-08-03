#!/usr/bin/env python3
"""
Test DirectML Setup for AMD RX 6800 XT
Simple verification that DirectML is working correctly
"""

import torch
import onnxruntime as ort
import sys

def test_directml_setup():
    """Test DirectML setup and compatibility"""
    print("="*60)
    print("DIRECTML SETUP TEST FOR AMD RX 6800 XT")
    print("="*60)
    
    # Test 1: PyTorch version and CUDA availability
    print(f"\n1. PyTorch Version: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    print(f"   MPS Available: {torch.backends.mps.is_available()}")
    
    # Test 2: ONNX Runtime providers
    print(f"\n2. ONNX Runtime Providers:")
    providers = ort.get_available_providers()
    for provider in providers:
        print(f"   - {provider}")
    
    # Test 3: DirectML specific test
    print(f"\n3. DirectML Test:")
    if 'DmlExecutionProvider' in providers:
        print("   ✓ DirectML provider is available!")
        
        # Test DirectML session
        try:
            # Create a simple test model
            import numpy as np
            
            # Simple matrix multiplication test
            a = np.random.randn(100, 100).astype(np.float32)
            b = np.random.randn(100, 100).astype(np.float32)
            
            # Test with DirectML
            session = ort.InferenceSession(
                None, 
                providers=['DmlExecutionProvider', 'CPUExecutionProvider']
            )
            print("   ✓ DirectML session created successfully!")
            
        except Exception as e:
            print(f"   ✗ DirectML session test failed: {e}")
    else:
        print("   ✗ DirectML provider not found!")
    
    # Test 4: PyTorch device test
    print(f"\n4. PyTorch Device Test:")
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"   Using MPS device: {device}")
        
        # Test tensor operations
        try:
            x = torch.randn(100, 100, device=device)
            y = torch.randn(100, 100, device=device)
            z = torch.mm(x, y)
            print("   ✓ MPS tensor operations successful!")
        except Exception as e:
            print(f"   ✗ MPS tensor operations failed: {e}")
    else:
        device = torch.device('cpu')
        print(f"   Using CPU device: {device}")
        
        # Test tensor operations
        try:
            x = torch.randn(100, 100, device=device)
            y = torch.randn(100, 100, device=device)
            z = torch.mm(x, y)
            print("   ✓ CPU tensor operations successful!")
        except Exception as e:
            print(f"   ✗ CPU tensor operations failed: {e}")
    
    # Test 5: Memory test
    print(f"\n5. Memory Test:")
    try:
        if torch.backends.mps.is_available():
            # Test GPU memory allocation
            large_tensor = torch.randn(1000, 1000, device='mps')
            print(f"   ✓ GPU memory allocation successful!")
            print(f"   Tensor size: {large_tensor.numel() * 4 / (1024*1024):.1f} MB")
        else:
            # Test CPU memory allocation
            large_tensor = torch.randn(1000, 1000, device='cpu')
            print(f"   ✓ CPU memory allocation successful!")
            print(f"   Tensor size: {large_tensor.numel() * 4 / (1024*1024):.1f} MB")
    except Exception as e:
        print(f"   ✗ Memory allocation failed: {e}")
    
    print("\n" + "="*60)
    print("SETUP TEST COMPLETED")
    print("="*60)

if __name__ == "__main__":
    test_directml_setup() 